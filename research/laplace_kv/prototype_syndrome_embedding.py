#!/usr/bin/env python3
"""Embed compact KVarN metadata in fixed-width K4/V2 code indices."""

import argparse
import math

import numpy as np

from prototype_delta import (
    load_trace,
    relative_error,
    softmax,
    stack,
    variance_normalize,
)
from prototype_kvarn_official import (
    even_transform_rows,
    half,
    store_even_transform_rows,
)


def append_bits(output, values, width):
    for value in np.asarray(values).ravel():
        value = int(value)
        output.extend((value >> bit) & 1 for bit in range(width))


def take_bits(bits, position, count, width):
    values = np.empty(count, dtype=np.uint16)
    for index in range(count):
        value = 0
        for bit in range(width):
            value |= int(bits[position]) << bit
            position += 1
        values[index] = value
    return values, position


def encode_field(values, bits, payload):
    values = np.asarray(values, dtype=np.float64).ravel()
    low = float(half(np.min(values)))
    maximum = (1 << bits) - 1
    step = float(half((np.max(values) - np.min(values)) / maximum))
    step = max(step, 2.0 ** -24)
    codes = np.clip(
        np.rint((values - low) / step), 0, maximum
    ).astype(np.uint16)
    endpoints = np.asarray([low, step], dtype="<f2").view("<u2")
    append_bits(payload, endpoints, 16)
    append_bits(payload, codes, bits)
    return codes.astype(np.float64) * step + low


def decode_fields(payload, lengths, bits):
    fields = []
    position = 0
    for length in lengths:
        endpoints, position = take_bits(payload, position, 2, 16)
        low, step = endpoints.astype("<u2").view("<f2").astype(np.float64)
        codes, position = take_bits(payload, position, length, bits)
        fields.append(codes.astype(np.float64) * step + low)
    if position != len(payload):
        raise RuntimeError("metadata payload length mismatch")
    return fields


def tile_components(source, bits, key, metadata_bits):
    rotated = store_even_transform_rows(source)
    oriented = rotated.T if key else rotated
    balanced, columns, rows = variance_normalize(oriented, iterations=8)
    low = np.min(balanced, axis=1, keepdims=True)
    high = np.max(balanced, axis=1, keepdims=True)
    step = np.maximum((high - low) / ((1 << bits) - 1), 1e-10)
    codes = np.ascontiguousarray(np.clip(
        np.rint((balanced - low) / step), 0, (1 << bits) - 1
    ).astype(np.uint8))
    payload = []
    fields = (
        encode_field((rows * step).ravel(), metadata_bits, payload),
        encode_field((rows * low).ravel(), metadata_bits, payload),
        encode_field(columns.ravel(), metadata_bits, payload),
    )
    return codes, fields, oriented, np.asarray(payload, dtype=np.uint8)


def reconstruct(codes, fields, key):
    absorbed_scale, absorbed_zero, column_scale = fields
    restored = (
        codes * absorbed_scale[:, None] + absorbed_zero[:, None]
    ) * column_scale[None, :]
    restored = restored.T if key else restored
    return half(even_transform_rows(restored)).astype(np.float32)


def syndrome(bits, power):
    value = 0
    for index, bit in enumerate(bits, 1):
        if bit:
            value ^= index
    return value


def embed_groups(codes, payload, power):
    """Embed payload with shortened Hamming matrix groups in one code array."""
    width = (1 << power) - 1
    groups = math.ceil(len(payload) / power)
    if groups * width > codes.size:
        raise ValueError("carrier capacity exceeded")
    flat = codes.ravel()
    for group in range(groups):
        start = group * width
        message = 0
        for bit in range(power):
            index = group * power + bit
            if index < len(payload):
                message |= int(payload[index]) << bit
        current = syndrome(flat[start:start + width] & 1, power)
        flip = current ^ message
        if flip:
            index = start + flip - 1
            flat[index] = int(flat[index]) ^ 1
    extracted = extract_groups(flat, len(payload), power)
    if not np.array_equal(extracted, payload):
        raise RuntimeError("Hamming metadata roundtrip failed")
    return groups * width


def extract_groups(codes, payload_bits, power):
    width = (1 << power) - 1
    output = []
    for start in range(0, math.ceil(payload_bits / power) * width, width):
        value = syndrome(codes[start:start + width] & 1, power)
        output.extend((value >> bit) & 1 for bit in range(power))
    return np.asarray(output[:payload_bits], dtype=np.uint8)


def neighbor_options(codes, fields, source, key):
    """Return the nearest opposite-parity code and its source-domain cost."""
    maximum = 15 if key else 3
    rows, columns = codes.shape
    absorbed_scale, absorbed_zero, column_scale = fields
    neighbors = np.empty(codes.size, dtype=np.uint8)
    costs = np.empty(codes.size, dtype=np.float64)
    for index, code in enumerate(codes.ravel()):
        row, column = divmod(index, columns)
        candidates = [value for value in (int(code) - 1, int(code) + 1)
                      if 0 <= value <= maximum]
        target = source[row, column]
        neighbor = min(
            candidates,
            key=lambda value: abs(
                (value * absorbed_scale[row] + absorbed_zero[row])
                * column_scale[column] - target
            ),
        )
        reconstructed = (
            neighbor * absorbed_scale[row] + absorbed_zero[row]
        ) * column_scale[column]
        neighbors[index] = neighbor
        costs[index] = (reconstructed - target) ** 2
    return neighbors, costs


def spread_blocks(size, groups):
    return [
        (group * size // groups, (group + 1) * size // groups)
        for group in range(groups)
    ]


def embed_spread(codes, payload, power, neighbors, costs):
    """Use repeated Hamming columns and select the cheapest valid flip."""
    width = (1 << power) - 1
    groups = math.ceil(len(payload) / power)
    if codes.size // groups < width:
        raise ValueError("spread carrier capacity exceeded")
    for group, (start, stop) in enumerate(spread_blocks(codes.size, groups)):
        labels = 1 + np.arange(stop - start) % width
        current = 0
        for label, bit in zip(labels, codes[start:stop] & 1):
            if bit:
                current ^= int(label)
        message = 0
        for bit in range(power):
            index = group * power + bit
            if index < len(payload):
                message |= int(payload[index]) << bit
        flip = current ^ message
        if flip:
            candidates = np.flatnonzero(labels == flip) + start
            index = int(candidates[np.argmin(costs[candidates])])
            codes[index] = neighbors[index]
    extracted = extract_spread(codes, len(payload), power)
    if not np.array_equal(extracted, payload):
        raise RuntimeError("spread metadata roundtrip failed")


def extract_spread(codes, payload_bits, power):
    if not payload_bits:
        return np.empty(0, dtype=np.uint8)
    width = (1 << power) - 1
    groups = math.ceil(payload_bits / power)
    output = []
    for start, stop in spread_blocks(codes.size, groups):
        value = 0
        for label, bit in enumerate(codes[start:stop] & 1, 1):
            if bit:
                value ^= 1 + (label - 1) % width
        output.extend((value >> bit) & 1 for bit in range(power))
    return np.asarray(output[:payload_bits], dtype=np.uint8)


def set_flip_direction(modified, original, fields, source, key):
    """Choose +1 or -1 parity neighbor nearest the unquantized source."""
    changed = np.flatnonzero(modified.ravel() != original.ravel())
    maximum = 15 if key else 3
    flat = modified.ravel()
    old = original.ravel()
    rows, columns = modified.shape
    absorbed_scale, absorbed_zero, column_scale = fields
    for index in changed:
        row, column = divmod(int(index), columns)
        candidates = [value for value in (int(old[index]) - 1,
                                           int(old[index]) + 1)
                      if 0 <= value <= maximum]
        target = source[row, column]
        flat[index] = min(
            candidates,
            key=lambda value: abs(
                (value * absorbed_scale[row] + absorbed_zero[row])
                * column_scale[column] - target
            ),
        )


def embed_tile(key_codes, value_codes, payload, scheme, key_fields,
               value_fields, key_source, value_source):
    original_k = key_codes.copy()
    original_v = value_codes.copy()
    key_flat = key_codes.ravel()
    value_flat = value_codes.ravel()
    key_neighbors, key_costs = neighbor_options(
        key_codes, key_fields, key_source, True
    )
    value_neighbors, value_costs = neighbor_options(
        value_codes, value_fields, value_source, False
    )

    if scheme in ("k_only_hamming", "k_only_spread"):
        power = max(
            power for power in range(2, 9)
            if (key_flat.size // ((1 << power) - 1)) * power
            >= len(payload)
        )
        if scheme == "k_only_hamming":
            embed_groups(key_flat, payload, power)
        else:
            embed_spread(
                key_flat, payload, power, key_neighbors, key_costs
            )
    elif scheme in ("k_first_h7", "k_first_spread"):
        power = 3
        width = 7
        key_bits = min(
            len(payload), (key_flat.size // width) * power
        )
        key_bits -= key_bits % power
        embed_groups(key_flat, payload[:key_bits], power)
        if scheme == "k_first_h7":
            embed_groups(value_flat, payload[key_bits:], power)
        elif key_bits < len(payload):
            embed_spread(
                value_flat, payload[key_bits:], power,
                value_neighbors, value_costs,
            )
    elif scheme in ("joint_hamming", "joint_spread"):
        carriers = np.concatenate((key_flat, value_flat))
        neighbors = np.concatenate((key_neighbors, value_neighbors))
        costs = np.concatenate((key_costs, value_costs))
        power = max(
            power for power in range(2, 9)
            if (carriers.size // ((1 << power) - 1)) * power
            >= len(payload)
        )
        if scheme == "joint_hamming":
            embed_groups(carriers, payload, power)
        else:
            embed_spread(carriers, payload, power, neighbors, costs)
        key_flat[:] = carriers[:key_flat.size]
        value_flat[:] = carriers[key_flat.size:]
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    set_flip_direction(
        key_codes, original_k, key_fields, key_source, True
    )
    set_flip_direction(
        value_codes, original_v, value_fields, value_source, False
    )

    # Direction selection preserves parity, so it must preserve the payload.
    if scheme in ("k_only_hamming", "k_only_spread"):
        extracted = (
            extract_groups(key_flat, len(payload), power)
            if scheme == "k_only_hamming"
            else extract_spread(key_flat, len(payload), power)
        )
    elif scheme in ("k_first_h7", "k_first_spread"):
        first = extract_groups(key_flat, key_bits, 3)
        if scheme == "k_first_h7":
            second = extract_groups(value_flat, len(payload) - key_bits, 3)
        else:
            second = extract_spread(value_flat, len(payload) - key_bits, 3)
        extracted = np.concatenate((first, second))
    else:
        carriers = np.concatenate((key_flat, value_flat))
        extracted = (
            extract_groups(carriers, len(payload), power)
            if scheme == "joint_hamming"
            else extract_spread(carriers, len(payload), power)
        )
    if not np.array_equal(extracted, payload):
        raise RuntimeError("direction selection corrupted metadata")
    return extracted, (
        int(np.count_nonzero(key_codes != original_k)),
        int(np.count_nonzero(value_codes != original_v)),
        power,
    )


def encode_pair(key_source, value_source, metadata_bits, scheme=None):
    key_codes, key_fields, key_oriented, key_payload = tile_components(
        key_source, 4, True, metadata_bits
    )
    value_codes, value_fields, value_oriented, value_payload = tile_components(
        value_source, 2, False, metadata_bits
    )
    payload = np.concatenate((key_payload, value_payload))
    lengths = (
        len(key_fields[0]), len(key_fields[1]), len(key_fields[2]),
        len(value_fields[0]), len(value_fields[1]), len(value_fields[2]),
    )
    decoded = decode_fields(payload, lengths, metadata_bits)
    key_fields = tuple(decoded[:3])
    value_fields = tuple(decoded[3:])
    flips = (0, 0, 0)
    if scheme:
        extracted, flips = embed_tile(
            key_codes, value_codes, payload, scheme,
            key_fields, value_fields, key_oriented, value_oriented,
        )
        decoded = decode_fields(extracted, lengths, metadata_bits)
        key_fields = tuple(decoded[:3])
        value_fields = tuple(decoded[3:])
    return (
        reconstruct(key_codes, key_fields, True),
        reconstruct(value_codes, value_fields, False),
        len(payload),
        flips,
    )


def self_test():
    for power in (2, 3, 4):
        width = (1 << power) - 1
        for message in range(1 << power):
            codes = np.arange(width, dtype=np.uint8) & 3
            payload = np.asarray(
                [(message >> bit) & 1 for bit in range(power)],
                dtype=np.uint8,
            )
            embed_groups(codes, payload, power)
            if not np.array_equal(extract_groups(codes, power, power), payload):
                raise RuntimeError("Hamming self-test failed")
            spread = np.arange(width * 3 + 2, dtype=np.uint8) & 3
            neighbors = spread ^ 1
            embed_spread(
                spread, payload, power, neighbors,
                np.arange(spread.size, dtype=np.float64),
            )
            if not np.array_equal(
                extract_spread(spread, power, power), payload
            ):
                raise RuntimeError("spread matrix self-test failed")


def summarize(values):
    values = np.asarray(values, dtype=np.float64) * 100.0
    return np.mean(values), np.percentile(values, 95), np.max(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--sink", type=int, default=128)
    args = parser.parse_args()
    self_test()
    keys, values, queries = load_trace(args.trace)
    dim = queries[0][6].size
    metadata_modes = (5, 6, 8)
    schemes = (
        None,
        "k_only_hamming", "k_only_spread",
        "k_first_h7", "k_first_spread",
        "joint_hamming", "joint_spread",
    )
    errors = {(bits, scheme): [] for bits in metadata_modes
              for scheme in schemes if bits != 8 or scheme is not None}
    flip_counts = {(bits, scheme): [] for bits in metadata_modes
                   for scheme in schemes if scheme is not None}
    cache = {}

    for layer, head, count, first, scale, _, query, _ in queries:
        identity = layer, head, count, first
        if identity not in cache:
            original_k = half(stack(keys, layer, head, count)[first:]).astype(
                np.float32
            )
            original_v = half(stack(values, layer, head, count)[first:]).astype(
                np.float32
            )
            tile = slice(args.sink, args.sink + args.group)
            if tile.stop > len(original_k):
                raise ValueError("trace does not contain one complete sealed tile")
            baseline_k = original_k.copy()
            baseline_v = original_v.copy()
            baseline_k[tile], baseline_v[tile], payload_bits, _ = encode_pair(
                original_k[tile], original_v[tile], 8
            )
            expected_payload = 3 * (dim + args.group) * 8 + 6 * 32
            if payload_bits != expected_payload:
                raise RuntimeError("unexpected Q8 metadata payload size")
            variants = {}
            for bits in metadata_modes:
                for scheme in schemes:
                    if bits == 8 and scheme is None:
                        continue
                    candidate_k = original_k.copy()
                    candidate_v = original_v.copy()
                    result_k, result_v, payload_bits, flips = encode_pair(
                        original_k[tile], original_v[tile], bits, scheme
                    )
                    expected_payload = (
                        3 * (dim + args.group) * bits + 6 * 32
                    )
                    if payload_bits != expected_payload:
                        raise RuntimeError("unexpected metadata payload size")
                    candidate_k[tile] = result_k
                    candidate_v[tile] = result_v
                    variants[bits, scheme] = candidate_k, candidate_v
                    if scheme:
                        flip_counts[bits, scheme].append(flips)
            cache[identity] = baseline_k, baseline_v, variants

        baseline_k, baseline_v, variants = cache[identity]
        baseline_weights = softmax(
            np.einsum("td,d->t", baseline_k, query) * scale
        )
        baseline = np.einsum("t,td->d", baseline_weights, baseline_v)
        for key, (candidate_k, candidate_v) in variants.items():
            weights = softmax(
                np.einsum("td,d->t", candidate_k, query) * scale
            )
            candidate = np.einsum("t,td->d", weights, candidate_v)
            errors[key].append(relative_error(candidate, baseline))

    scalar_count = 2 * dim * args.group
    fixed_bits = 6 * dim * args.group
    print(
        f"traces={len(queries)} tiles={len(cache)} dim={dim} "
        f"group={args.group} sink={args.sink}"
    )
    print(
        f"stored_bits={fixed_bits} scalars={scalar_count} "
        f"effective_bits={fixed_bits / scalar_count:.6f} "
        "external_metadata_bits=0"
    )
    print("errors are relative to unembedded K4/V2 with Q8 metadata")
    print("meta scheme       payload  power  K flips  V flips   mean%    p95%    max%")
    for bits in metadata_modes:
        for scheme in schemes:
            key = bits, scheme
            if key not in errors:
                continue
            label = "unembedded" if scheme is None else scheme
            mean, p95, maximum = summarize(errors[key])
            if scheme:
                flips = np.asarray(flip_counts[key])
                powers = sorted(set(int(value) for value in flips[:, 2]))
                power = "/".join(map(str, powers))
                key_flips = np.mean(flips[:, 0])
                value_flips = np.mean(flips[:, 1])
            else:
                power = "-"
                key_flips = value_flips = 0.0
            payload = 3 * (dim + args.group) * bits + 6 * 32
            print(
                f"Q{bits:<3d} {label:12s} {payload:7d} {power:>6s} "
                f"{key_flips:8.2f} {value_flips:8.2f} "
                f"{mean:8.3f} {p95:8.3f} {maximum:8.3f}"
            )


if __name__ == "__main__":
    main()

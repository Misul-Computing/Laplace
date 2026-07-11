#!/usr/bin/env python3
"""Test fixed-width joint K/V mixed-radix tiles on captured attention."""

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
    scale_field,
    store_even_transform_rows,
)


def rounded_codes(values, levels, mode, generator):
    clipped = np.clip(values, 0, levels - 1)
    if mode == "nearest":
        return np.rint(clipped)
    lower = np.floor(clipped)
    fraction = clipped - lower
    if mode == "independent":
        return lower + (generator.random(clipped.shape) < fraction)
    cumulative = np.cumsum(fraction, axis=0)
    previous = np.vstack((np.zeros((1, clipped.shape[1])), cumulative[:-1]))
    offset = generator.random((1, clipped.shape[1]))
    increments = np.floor(cumulative + offset) - np.floor(previous + offset)
    return lower + increments


def quantize_tile(
    source, levels, key, metadata_bits, rounding="nearest", generator=None
):
    rotated = store_even_transform_rows(source)
    oriented = rotated.T if key else rotated
    balanced, columns, rows = variance_normalize(oriented, iterations=8)
    low = np.min(balanced, axis=1, keepdims=True)
    high = np.max(balanced, axis=1, keepdims=True)
    step = np.maximum((high - low) / (levels - 1), 1e-10)
    normalized = (balanced - low) / step
    if generator is None:
        generator = np.random.default_rng(0)
    codes = rounded_codes(normalized, levels, rounding, generator)
    absorbed_scale = scale_field((rows * step).ravel(), metadata_bits)
    absorbed_zero = scale_field((rows * low).ravel(), metadata_bits)
    other_scale = scale_field(columns.ravel(), metadata_bits)
    restored = (
        codes * absorbed_scale[:, None] + absorbed_zero[:, None]
    ) * other_scale[None, :]
    restored = restored.T if key else restored
    return (
        half(even_transform_rows(restored)).astype(np.float32),
        codes.astype(np.uint8),
    )


def tile_bytes(
    dimension,
    group,
    key_levels,
    value_levels,
    key_metadata_bits,
    value_metadata_bits,
):
    pair_states = key_levels * value_levels
    pairs_per_word = 2
    word_bits = math.ceil(math.log2(pair_states ** pairs_per_word))
    pair_count = dimension * group
    code_bits = math.ceil(pair_count / pairs_per_word) * word_bits
    metadata_bits = (
        (2 * dimension + group) * key_metadata_bits
        + (2 * group + dimension) * value_metadata_bits
    )
    metadata = math.ceil(metadata_bits / 8) + 24
    raw = math.ceil(code_bits / 8) + metadata
    aligned = math.ceil(raw / 64) * 64
    scalars = 2 * dimension * group
    return word_bits, code_bits, metadata, raw, aligned, aligned * 8 / scalars


def split_k12_v4_bytes(
    dimension, group, key_metadata_bits, value_metadata_bits
):
    elements = dimension * group
    key_words = math.ceil(elements / 5)
    key_bits = key_words * 18
    value_bits = elements * 2
    metadata_bits = (
        (2 * dimension + group) * key_metadata_bits
        + (2 * group + dimension) * value_metadata_bits
        + 128
    )
    code_bytes = math.ceil(key_bits / 8) + math.ceil(value_bits / 8)
    raw = code_bytes + math.ceil(metadata_bits / 8)
    aligned = math.ceil(raw / 64) * 64
    scalars = 2 * elements
    return (
        18,
        key_bits + value_bits,
        math.ceil(metadata_bits / 8),
        raw,
        aligned,
        aligned * 8 / scalars,
    )


def pack_words(words, width):
    output = bytearray(math.ceil(len(words) * width / 8))
    accumulator = 0
    available = 0
    position = 0
    for word in words:
        accumulator |= int(word) << available
        available += width
        while available >= 8:
            output[position] = accumulator & 0xFF
            accumulator >>= 8
            available -= 8
            position += 1
    if available:
        output[position] = accumulator & 0xFF
    return bytes(output)


def unpack_words(payload, count, width):
    if width > 32:
        dtype = np.uint64
    elif width > 16:
        dtype = np.uint32
    else:
        dtype = np.uint16
    output = np.empty(count, dtype=dtype)
    accumulator = 0
    available = 0
    position = 0
    for index in range(count):
        while available < width:
            accumulator |= payload[position] << available
            available += 8
            position += 1
        output[index] = accumulator & ((1 << width) - 1)
        accumulator >>= width
        available -= width
    return output


def verify_joint_record(key_codes, value_codes, key_levels, value_levels):
    key = key_codes.T.ravel().astype(np.uint16)
    value = value_codes.ravel().astype(np.uint16)
    states = key + key_levels * value
    radix = key_levels * value_levels
    if len(states) % 2:
        states = np.append(states, 0)
    words = states[0::2] + radix * states[1::2]
    width = math.ceil(math.log2(radix * radix))
    payload = pack_words(words, width)
    decoded_words = unpack_words(payload, len(words), width)
    first = decoded_words % radix
    second = decoded_words // radix
    decoded_states = np.empty(len(states), dtype=np.uint16)
    decoded_states[0::2] = first
    decoded_states[1::2] = second
    decoded_key = decoded_states[:len(key)] % key_levels
    decoded_value = decoded_states[:len(value)] // key_levels
    if not np.array_equal(decoded_key, key):
        raise RuntimeError("joint key roundtrip failed")
    if not np.array_equal(decoded_value, value):
        raise RuntimeError("joint value roundtrip failed")
    return len(payload)


def verify_split_k12_v4_record(key_codes, value_codes):
    key = key_codes.ravel().astype(np.uint16)
    padded = np.pad(key, (0, (-len(key)) % 5))
    powers = np.asarray([1, 12, 144, 1728, 20736], dtype=np.uint32)
    words = np.sum(padded.reshape(-1, 5) * powers, axis=1, dtype=np.uint32)
    key_payload = pack_words(words, 18)
    decoded_words = unpack_words(key_payload, len(words), 18).astype(np.uint32)
    decoded_key = np.empty(len(padded), dtype=np.uint16)
    for index in range(5):
        decoded_key[index::5] = decoded_words % 12
        decoded_words //= 12
    if not np.array_equal(decoded_key[:len(key)], key):
        raise RuntimeError("split K12 roundtrip failed")

    value = value_codes.ravel().astype(np.uint16)
    value_payload = pack_words(value, 2)
    decoded_value = unpack_words(value_payload, len(value), 2)
    if not np.array_equal(decoded_value, value):
        raise RuntimeError("split V4 roundtrip failed")
    return len(key_payload) + len(value_payload)


def attention(keys, values, query, scale):
    weights = softmax(np.einsum("td,d->t", keys, query) * scale)
    return np.einsum("t,td->d", weights, values)


def summarize(label, values):
    values = np.asarray(values) * 100.0
    print(
        f"{label:18s} mean={np.mean(values):8.3f}% "
        f"p95={np.percentile(values, 95):8.3f}% max={np.max(values):8.3f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--sink", type=int, default=128)
    parser.add_argument("--key-levels", type=int, default=15)
    parser.add_argument("--value-levels", type=int, default=3)
    parser.add_argument("--metadata-bits", type=int, default=6)
    parser.add_argument("--key-metadata-bits", type=int)
    parser.add_argument("--value-metadata-bits", type=int)
    parser.add_argument("--split-k12-v4", action="store_true")
    parser.add_argument(
        "--value-rounding",
        choices=("nearest", "independent", "balanced"),
        default="nearest",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    context = max(item[2] - item[3] for item in queries)
    queries = [item for item in queries if item[2] - item[3] == context]
    dimension = queries[0][6].size
    key_metadata_bits = (
        args.metadata_bits
        if args.key_metadata_bits is None else args.key_metadata_bits
    )
    value_metadata_bits = (
        args.metadata_bits
        if args.value_metadata_bits is None else args.value_metadata_bits
    )

    if args.split_k12_v4:
        if args.key_levels != 12 or args.value_levels != 4:
            raise ValueError("split layout requires K12/V4")
        storage = split_k12_v4_bytes(
            dimension, args.group, key_metadata_bits, value_metadata_bits
        )
    else:
        storage = tile_bytes(
            dimension,
            args.group,
            args.key_levels,
            args.value_levels,
            key_metadata_bits,
            value_metadata_bits,
        )
    word_bits, code_bits, metadata, raw, aligned, effective = storage
    layout = "split-K5x18/V2" if args.split_k12_v4 else "joint-pair2"
    print(
        f"K{args.key_levels}/V{args.value_levels} layout={layout} "
        f"word_bits={word_bits} code_bits={code_bits} metadata_bytes={metadata} "
        f"raw_bytes={raw} aligned_bytes={aligned} effective_bits={effective:.6f}"
    )

    labels = ("candidate", "candidate exact V", "exact K candidate V", "K16/V4 Q8")
    errors = {label: [] for label in labels}
    cache = {}
    for layer, head, count, first, scale, _, query, _ in queries:
        identity = layer, head, count, first
        if identity not in cache:
            original_k = stack(keys, layer, head, count)[first:].astype(np.float64)
            original_v = stack(values, layer, head, count)[first:].astype(np.float64)
            tile = slice(args.sink, args.sink + args.group)
            if tile.stop > len(original_k):
                raise ValueError("trace does not contain one sealed tile")
            candidate_k = original_k.copy()
            candidate_v = original_v.copy()
            candidate_k[tile], candidate_key_codes = quantize_tile(
                original_k[tile], args.key_levels, True, key_metadata_bits
            )
            candidate_v[tile], candidate_value_codes = quantize_tile(
                original_v[tile],
                args.value_levels,
                False,
                value_metadata_bits,
                args.value_rounding,
                np.random.default_rng(np.random.SeedSequence((
                    args.seed, layer, head, args.sink,
                ))),
            )
            if args.split_k12_v4:
                packed_bytes = verify_split_k12_v4_record(
                    candidate_key_codes, candidate_value_codes
                )
            else:
                packed_bytes = verify_joint_record(
                    candidate_key_codes,
                    candidate_value_codes,
                    args.key_levels,
                    args.value_levels,
                )
            if packed_bytes != raw - metadata:
                raise RuntimeError("joint record size mismatch")
            control_k = original_k.copy()
            control_v = original_v.copy()
            control_k[tile], _ = quantize_tile(original_k[tile], 16, True, 8)
            control_v[tile], _ = quantize_tile(original_v[tile], 4, False, 8)
            cache[identity] = (
                original_k,
                original_v,
                candidate_k,
                candidate_v,
                control_k,
                control_v,
            )
        (original_k, original_v, candidate_k, candidate_v,
         control_k, control_v) = cache[identity]
        query = query.astype(np.float64)
        reference = attention(original_k, original_v, query, scale)
        outputs = {
            "candidate": attention(candidate_k, candidate_v, query, scale),
            "candidate exact V": attention(candidate_k, original_v, query, scale),
            "exact K candidate V": attention(original_k, candidate_v, query, scale),
            "K16/V4 Q8": attention(control_k, control_v, query, scale),
        }
        for label, output in outputs.items():
            errors[label].append(relative_error(output, reference))
    for label in labels:
        summarize(label, errors[label])


if __name__ == "__main__":
    main()

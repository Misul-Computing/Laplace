#!/usr/bin/env python3
"""Enumerate the fully packed scalar-alphabet frontier at three bits."""

import argparse
import math

import numpy as np

from prototype_delta import load_trace, relative_error, stack
from prototype_kvarn_official import half
from prototype_mixed_radix import (
    attention,
    pack_words,
    quantize_tile,
    unpack_words,
)


TILE = 128
HEADER_BITS = 128


def packed_bytes(elements, levels):
    best = None
    for group in range(1, 65):
        width = math.ceil(group * math.log2(levels))
        if width > 64:
            break
        bits = math.ceil(elements / group) * width
        candidate = math.ceil(bits / 8), group, width
        if best is None or candidate < best:
            best = candidate
    return best


def verify_codes(codes, levels, group, width):
    source = codes.ravel().astype(np.uint64)
    padded = np.pad(source, (0, (-len(source)) % group))
    powers = np.asarray([levels ** index for index in range(group)], dtype=np.uint64)
    words = np.sum(
        padded.reshape(-1, group) * powers, axis=1, dtype=np.uint64
    )
    payload = pack_words(words, width)
    decoded_words = unpack_words(payload, len(words), width).astype(np.uint64)
    decoded = np.empty(len(padded), dtype=np.uint16)
    for index in range(group):
        decoded[index::group] = decoded_words % levels
        decoded_words //= levels
    if not np.array_equal(decoded[:len(source)], source):
        raise RuntimeError("alphabet stream roundtrip failed")
    return len(payload)


def layout(dimension, tokens, key_levels, value_levels, key_meta, value_meta):
    elements = dimension * tokens
    key_bytes, key_group, key_width = packed_bytes(elements, key_levels)
    value_bytes, value_group, value_width = packed_bytes(elements, value_levels)
    metadata_bits = (
        (2 * dimension + tokens) * key_meta
        + (2 * tokens + dimension) * value_meta
        + HEADER_BITS
    )
    raw = key_bytes + value_bytes + math.ceil(metadata_bits / 8)
    aligned = math.ceil(raw / 64) * 64
    rate = aligned * 8 / (2 * elements)
    return rate, raw, aligned, key_group, key_width, value_group, value_width


def frontier(dimension, tokens):
    feasible = []
    for key_levels in range(3, 17):
        for value_levels in range(2, 9):
            for key_meta in range(2, 9):
                for value_meta in range(2, 9):
                    record = layout(
                        dimension,
                        tokens,
                        key_levels,
                        value_levels,
                        key_meta,
                        value_meta,
                    )
                    if record[0] <= 3.0:
                        feasible.append((
                            key_levels, value_levels, key_meta, value_meta,
                            *record,
                        ))
    output = []
    for candidate in feasible:
        quality = candidate[:4]
        dominated = any(
            all(other[index] >= quality[index] for index in range(4))
            and other[:4] != quality
            for other in feasible
        )
        if not dominated:
            output.append(candidate)
    return output


def summarize(values):
    values = np.asarray(values) * 100.0
    return np.mean(values), np.percentile(values, 95), np.max(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--candidate", type=int, nargs=4)
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    context = max(item[2] - item[3] for item in queries)
    queries = [item for item in queries if item[2] - item[3] == context]
    dimension = queries[0][6].size
    if context < 2 * TILE:
        raise ValueError("trace needs at least 256 tokens")

    identities = {}
    records = []
    for layer, head, count, first, scale, _, query, _ in queries:
        identity = layer, head, count, first
        if identity not in identities:
            identities[identity] = (
                half(stack(keys, layer, head, count)[first:]),
                half(stack(values, layer, head, count)[first:]),
            )
        original_k, original_v = identities[identity]
        records.append((
            identity,
            scale,
            query.astype(np.float64),
            attention(original_k, original_v, query, scale),
        ))

    region = slice(TILE, 2 * TILE)
    controls = {}
    for identity, (original_k, original_v) in identities.items():
        candidate_k = original_k.copy()
        candidate_v = original_v.copy()
        candidate_k[region], _ = quantize_tile(
            original_k[region], 16, True, 8
        )
        candidate_v[region], _ = quantize_tile(
            original_v[region], 4, False, 8
        )
        controls[identity] = candidate_k, candidate_v
    control_errors = [
        relative_error(
            attention(*controls[identity], query, scale), reference
        )
        for identity, scale, query, reference in records
    ]
    control = summarize(control_errors)

    if args.candidate:
        key_levels, value_levels, key_meta, value_meta = args.candidate
        record = layout(
            dimension, TILE, key_levels, value_levels, key_meta, value_meta
        )
        if record[0] > 3.0:
            raise ValueError("candidate exceeds three bits")
        candidates = [(key_levels, value_levels, key_meta, value_meta, *record)]
    else:
        candidates = frontier(dimension, TILE)
    results = []
    for candidate in candidates:
        key_levels, value_levels, key_meta, value_meta = candidate[:4]
        variants = {}
        for identity, (original_k, original_v) in identities.items():
            candidate_k = original_k.copy()
            candidate_v = original_v.copy()
            candidate_k[region], key_codes = quantize_tile(
                original_k[region], key_levels, True, key_meta
            )
            candidate_v[region], value_codes = quantize_tile(
                original_v[region], value_levels, False, value_meta
            )
            if args.candidate:
                key_bytes = verify_codes(
                    key_codes, key_levels, candidate[7], candidate[8]
                )
                value_bytes = verify_codes(
                    value_codes, value_levels, candidate[9], candidate[10]
                )
                metadata_bits = (
                    (2 * dimension + TILE) * key_meta
                    + (2 * TILE + dimension) * value_meta
                    + HEADER_BITS
                )
                if key_bytes + value_bytes + math.ceil(metadata_bits / 8) != candidate[5]:
                    raise RuntimeError("candidate byte count mismatch")
            variants[identity] = candidate_k, candidate_v
        errors = [
            relative_error(
                attention(*variants[identity], query, scale), reference
            )
            for identity, scale, query, reference in records
        ]
        results.append((*summarize(errors), candidate))

    results.sort(key=lambda item: (item[0], item[1]))
    print(
        f"trace={args.trace} queries={len(queries)} D={dimension} "
        f"frontier={len(candidates)} exact_group_pack<=64bits"
    )
    print(
        f"control K16/V4-Q8 mean={control[0]:.3f}% "
        f"p95={control[1]:.3f}% max={control[2]:.3f}%"
    )
    for mean, p95, maximum, candidate in results[:args.top]:
        kl, vl, km, vm, rate, raw, aligned, kg, kw, vg, vw = candidate
        print(
            f"K{kl}/V{vl} Q{km}/Q{vm} rate={rate:.6f} "
            f"raw={raw} aligned={aligned} Kpack={kg}x/{kw}b "
            f"Vpack={vg}x/{vw}b mean={mean:.3f}% "
            f"p95={p95:.3f}% max={maximum:.3f}%"
        )


if __name__ == "__main__":
    main()

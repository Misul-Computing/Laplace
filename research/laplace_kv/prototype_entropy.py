#!/usr/bin/env python3
"""Measure lossless entropy bounds for normalized K4/V2 tiles."""

import argparse
import math

import numpy as np

from prototype_delta import load_trace, stack, variance_normalize
from prototype_kvarn_official import scale_field, store_hadamard_rows


RANS_BITS = 12
RANS_TOTAL = 1 << RANS_BITS
RANS_LOW = 1 << 23


def encode_tile(source, bits, key):
    rotated = store_hadamard_rows(source)
    oriented = rotated.T if key else rotated
    balanced, columns, rows = variance_normalize(oriented, iterations=8)
    low = np.min(balanced, axis=1, keepdims=True)
    high = np.max(balanced, axis=1, keepdims=True)
    step = np.maximum((high - low) / ((1 << bits) - 1), 1e-10)
    codes = np.clip(
        np.rint((balanced - low) / step), 0, (1 << bits) - 1
    ).astype(np.uint8)
    metadata = (
        scale_field((rows * step).ravel()),
        scale_field((rows * low).ravel()),
        scale_field(columns.ravel()),
    )
    return codes, metadata


def entropy_bits(codes, levels):
    counts = np.bincount(codes.ravel(), minlength=levels)
    nonzero = counts[counts > 0]
    probabilities = nonzero / np.sum(nonzero)
    return float(-np.sum(nonzero * np.log2(probabilities)))


def normalized_frequencies(symbols, levels):
    counts = np.bincount(symbols, minlength=levels)
    exact = counts * (RANS_TOTAL / max(1, np.sum(counts)))
    frequencies = np.floor(exact).astype(np.int64)
    frequencies[(counts > 0) & (frequencies == 0)] = 1
    while np.sum(frequencies) < RANS_TOTAL:
        candidates = np.where(counts > 0)[0]
        index = candidates[np.argmax(exact[candidates] - frequencies[candidates])]
        frequencies[index] += 1
    while np.sum(frequencies) > RANS_TOTAL:
        candidates = np.where(frequencies > 1)[0]
        index = candidates[np.argmax(frequencies[candidates] - exact[candidates])]
        frequencies[index] -= 1
    cumulative = np.zeros(levels, dtype=np.int64)
    cumulative[1:] = np.cumsum(frequencies[:-1])
    return frequencies, cumulative


def rans_encode(codes, levels):
    symbols = codes.ravel().astype(np.int64)
    frequencies, cumulative = normalized_frequencies(symbols, levels)
    state = RANS_LOW
    emitted = []
    for symbol in symbols[::-1]:
        frequency = int(frequencies[symbol])
        threshold = ((RANS_LOW >> RANS_BITS) << 8) * frequency
        while state >= threshold:
            emitted.append(state & 255)
            state >>= 8
        quotient, remainder = divmod(state, frequency)
        state = (quotient << RANS_BITS) + remainder
        state += int(cumulative[symbol])
    payload = state.to_bytes(4, "little") + bytes(reversed(emitted))

    table = np.empty(RANS_TOTAL, dtype=np.int16)
    for symbol, frequency in enumerate(frequencies):
        start = cumulative[symbol]
        table[start:start + frequency] = symbol
    decoded = np.empty_like(symbols)
    state = int.from_bytes(payload[:4], "little")
    position = 4
    for index in range(symbols.size):
        slot = state & (RANS_TOTAL - 1)
        symbol = int(table[slot])
        decoded[index] = symbol
        state = int(frequencies[symbol]) * (state >> RANS_BITS)
        state += slot - int(cumulative[symbol])
        while state < RANS_LOW and position < len(payload):
            state = (state << 8) | payload[position]
            position += 1
    if not np.array_equal(decoded, symbols):
        raise RuntimeError("rANS roundtrip failed")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--sink", type=int, default=128)
    parser.add_argument("--metadata-bits", type=int, default=5)
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    seen = set()
    tile_rates = []
    key_entropies = []
    value_entropies = []
    dim = queries[0][6].size
    histogram_bytes = (16 + 4) * 2
    header_bytes = 8
    metadata_bytes = (
        math.ceil(3 * (dim + args.group) * args.metadata_bits / 8) + 24
    )

    for layer, head, count, first, _, _, _, _ in queries:
        identity = layer, head, count, first
        if identity in seen:
            continue
        seen.add(identity)
        key_matrix = stack(keys, layer, head, count)[first:]
        value_matrix = stack(values, layer, head, count)[first:]
        for start in range(args.sink, count - args.group + 1, args.group):
            key_codes, _ = encode_tile(
                key_matrix[start:start + args.group], 4, True
            )
            value_codes, _ = encode_tile(
                value_matrix[start:start + args.group], 2, False
            )
            key_bits = entropy_bits(key_codes, 16)
            value_bits = entropy_bits(value_codes, 4)
            key_entropies.append(key_bits / key_codes.size)
            value_entropies.append(value_bits / value_codes.size)
            key_payload = rans_encode(key_codes, 16)
            value_payload = rans_encode(value_codes, 4)
            total_bytes = (
                len(key_payload) + len(value_payload)
                + metadata_bytes + histogram_bytes + header_bytes
            )
            total_bytes = (total_bytes + 15) // 16 * 16
            tile_rates.append(total_bytes * 8.0 / (2 * dim * args.group))

    print(
        f"tiles={len(tile_rates)} dim={dim} group={args.group} "
        f"metadata_bytes={metadata_bytes} histogram_bytes={histogram_bytes} "
        f"header_bytes={header_bytes} alignment=16"
    )
    for name, values_ in (
        ("K4 entropy", key_entropies),
        ("V2 entropy", value_entropies),
        ("total effective", tile_rates),
    ):
        values_ = np.asarray(values_)
        print(
            f"{name}: mean={np.mean(values_):.6f} "
            f"p95={np.percentile(values_, 95):.6f} max={np.max(values_):.6f}"
        )


if __name__ == "__main__":
    main()

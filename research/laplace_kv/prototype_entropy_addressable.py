#!/usr/bin/env python3
"""Count an independently addressable four-lane rANS K4/V2 tile."""

import argparse
import math

import numpy as np

from prototype_delta import load_trace, stack
from prototype_entropy import (
    RANS_BITS,
    RANS_LOW,
    encode_tile,
    normalized_frequencies,
)


def rans_encode_four(codes, levels):
    symbols = codes.ravel().astype(np.int64)
    frequencies, cumulative = normalized_frequencies(symbols, levels)
    states = [RANS_LOW] * 4
    emitted = []
    for index in range(symbols.size - 1, -1, -1):
        lane = index & 3
        symbol = int(symbols[index])
        frequency = int(frequencies[symbol])
        threshold = ((RANS_LOW >> RANS_BITS) << 8) * frequency
        state = states[lane]
        while state >= threshold:
            emitted.append(state & 255)
            state >>= 8
        quotient, remainder = divmod(state, frequency)
        states[lane] = (
            (quotient << RANS_BITS) + remainder + int(cumulative[symbol])
        )
    payload = b"".join(state.to_bytes(4, "little") for state in states)
    payload += bytes(reversed(emitted))

    table = np.empty(1 << RANS_BITS, dtype=np.uint8)
    for symbol, frequency in enumerate(frequencies):
        start = int(cumulative[symbol])
        table[start:start + frequency] = symbol
    decoded = np.empty_like(symbols)
    states = [
        int.from_bytes(payload[lane * 4:lane * 4 + 4], "little")
        for lane in range(4)
    ]
    position = 16
    for index in range(symbols.size):
        lane = index & 3
        state = states[lane]
        slot = state & ((1 << RANS_BITS) - 1)
        symbol = int(table[slot])
        decoded[index] = symbol
        state = int(frequencies[symbol]) * (state >> RANS_BITS)
        state += slot - int(cumulative[symbol])
        while state < RANS_LOW and position < len(payload):
            state = (state << 8) | payload[position]
            position += 1
        states[lane] = state
    if not np.array_equal(decoded, symbols) or position != len(payload):
        raise RuntimeError("four-lane rANS roundtrip failed")
    return payload


def context_rans_encode_four(codes, levels, context_shift):
    values = np.asarray(codes, dtype=np.uint8)
    rows, width = values.shape
    contexts = np.zeros_like(values)
    contexts[1:] = values[:-1] >> context_shift
    context_count = ((levels - 1) >> context_shift) + 1
    frequencies = []
    cumulative = []
    for context in range(context_count):
        symbols = values[contexts == context].ravel().astype(np.int64)
        if symbols.size == 0:
            symbols = np.arange(levels, dtype=np.int64)
        frequency, starts = normalized_frequencies(symbols, levels)
        frequencies.append(frequency)
        cumulative.append(starts)

    symbols = values.ravel().astype(np.int64)
    model_ids = contexts.ravel().astype(np.int64)
    states = [RANS_LOW] * 4
    emitted = []
    for index in range(symbols.size - 1, -1, -1):
        lane = index & 3
        symbol = int(symbols[index])
        model = int(model_ids[index])
        frequency = int(frequencies[model][symbol])
        threshold = ((RANS_LOW >> RANS_BITS) << 8) * frequency
        state = states[lane]
        while state >= threshold:
            emitted.append(state & 255)
            state >>= 8
        quotient, remainder = divmod(state, frequency)
        states[lane] = (
            (quotient << RANS_BITS) + remainder
            + int(cumulative[model][symbol])
        )
    payload = b"".join(state.to_bytes(4, "little") for state in states)
    payload += bytes(reversed(emitted))

    tables = []
    for model in range(context_count):
        table = np.empty(1 << RANS_BITS, dtype=np.uint8)
        for symbol, frequency in enumerate(frequencies[model]):
            start = int(cumulative[model][symbol])
            table[start:start + frequency] = symbol
        tables.append(table)
    decoded = np.empty_like(symbols)
    states = [
        int.from_bytes(payload[lane * 4:lane * 4 + 4], "little")
        for lane in range(4)
    ]
    position = 16
    for index in range(symbols.size):
        lane = index & 3
        row, column = divmod(index, width)
        previous = 0 if row == 0 else int(decoded[index - width])
        model = previous >> context_shift
        state = states[lane]
        slot = state & ((1 << RANS_BITS) - 1)
        symbol = int(tables[model][slot])
        decoded[index] = symbol
        state = int(frequencies[model][symbol]) * (state >> RANS_BITS)
        state += slot - int(cumulative[model][symbol])
        while state < RANS_LOW and position < len(payload):
            state = (state << 8) | payload[position]
            position += 1
        states[lane] = state
    if not np.array_equal(decoded, symbols) or position != len(payload):
        raise RuntimeError("four-lane context rANS roundtrip failed")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--sink", type=int, default=128)
    parser.add_argument("--alignment", type=int, default=4)
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    dim = queries[0][6].size

    # Six low/step FP16 pairs plus one continuous five-bit metadata field.
    metadata_bytes = math.ceil(3 * (dim + args.group) * 5 / 8) + 24
    frequency_bytes = (15 + 3) * 2  # Final frequency is implied per alphabet.
    header_bytes = 4                 # uint16 K length, uint16 V length.
    index_bytes = 8                  # Absolute archive offset for this tile.
    rates = []
    payload_rates = []
    context_rates_4 = []
    context_rates_16 = []
    seen = set()
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
            payload_bytes = len(rans_encode_four(key_codes.T, 16))
            payload_bytes += len(rans_encode_four(value_codes.T, 4))
            total_bytes = (
                payload_bytes + metadata_bytes + frequency_bytes
                + header_bytes + index_bytes
            )
            total_bytes = (
                (total_bytes + args.alignment - 1) // args.alignment
                * args.alignment
            )
            scalars = 2 * dim * args.group
            payload_rates.append(payload_bytes * 8.0 / scalars)
            rates.append(total_bytes * 8.0 / scalars)

            context_payload = len(context_rans_encode_four(
                key_codes.T, 16, 2
            ))
            context_payload += len(context_rans_encode_four(
                value_codes, 4, 0
            ))
            context_record = context_payload + 600 + 144 + 8
            context_rates_4.append(
                (((context_record + 3) // 4 * 4) + 8) * 8.0 / scalars
            )
            context_rates_16.append(
                (((context_record + 15) // 16 * 16) + 8) * 8.0 / scalars
            )

    print(
        f"tiles={len(rates)} dim={dim} group={args.group} "
        f"metadata_bytes={metadata_bytes} frequency_bytes={frequency_bytes} "
        f"header_bytes={header_bytes} index_bytes={index_bytes} "
        f"alignment={args.alignment}"
    )
    for name, values in (("payload", payload_rates), ("total", rates)):
        values = np.asarray(values)
        print(
            f"{name}: mean={np.mean(values):.6f} "
            f"p95={np.percentile(values, 95):.6f} max={np.max(values):.6f}"
        )
    for alignment, rates_ in ((4, context_rates_4), (16, context_rates_16)):
        values = np.asarray(rates_)
        print(
            f"context-q8-addressable-a{alignment}: "
            f"mean={np.mean(values):.6f} "
            f"p95={np.percentile(values, 95):.6f} "
            f"max={np.max(values):.6f}"
        )


if __name__ == "__main__":
    main()

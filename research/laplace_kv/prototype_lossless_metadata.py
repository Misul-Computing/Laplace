#!/usr/bin/env python3
"""Count reversible K4/V2 code and FP16-metadata tile codecs."""

import argparse
import bz2
import ctypes
import lzma
import math
import zlib

import numpy as np

from prototype_delta import load_trace, stack, variance_normalize
from prototype_entropy import (
    RANS_BITS,
    RANS_LOW,
    RANS_TOTAL,
    normalized_frequencies,
    rans_encode,
)
from prototype_kvarn_official import half, store_hadamard_rows


COMPRESSION_LZ4 = 0x100
COMPRESSION_ZLIB = 0x205
COMPRESSION_LZFSE = 0x801


def official_tile(source, bits, key):
    rotated = store_hadamard_rows(source)
    oriented = rotated.T if key else rotated
    balanced, columns, rows = variance_normalize(oriented, iterations=8)
    low = np.min(balanced, axis=1, keepdims=True)
    high = np.max(balanced, axis=1, keepdims=True)
    step = np.maximum((high - low) / ((1 << bits) - 1), 1e-10)
    codes = np.clip(
        np.rint((balanced - low) / step), 0, (1 << bits) - 1
    ).astype(np.uint8)
    fields = tuple(
        np.asarray(half(field.ravel()), dtype="<f2").view("<u2")
        for field in (rows * step, rows * low, columns)
    )
    return codes, fields


def pack_codes(codes, bits):
    symbols = codes.ravel().astype(np.uint8)
    output = bytearray(math.ceil(symbols.size * bits / 8))
    bit = 0
    for symbol in symbols:
        byte = bit >> 3
        shift = bit & 7
        output[byte] |= int(symbol) << shift
        if shift + bits > 8:
            output[byte + 1] |= int(symbol) >> (8 - shift)
        bit += bits
    return bytes(output)


def predictor(codes, axis, mode):
    values = np.asarray(codes, dtype=np.uint8)
    levels = int(np.max(values)) + 1
    if axis == "flat":
        values = values.ravel()
        previous = np.empty_like(values)
        previous[0] = 0
        previous[1:] = values[:-1]
    else:
        previous = np.roll(values, 1, axis=axis)
        selector = [slice(None)] * values.ndim
        selector[axis] = 0
        previous[tuple(selector)] = 0
    if mode == "delta":
        encoded = (values.astype(np.int16) - previous) % levels
    elif mode == "xor":
        encoded = values ^ previous
    else:
        encoded = values
    return encoded.astype(np.uint8).ravel()


def field_predict(fields, mode):
    output = []
    for field in fields:
        values = field.astype(np.uint16)
        previous = np.empty_like(values)
        previous[0] = 0
        previous[1:] = values[:-1]
        if mode == "xor":
            encoded = values ^ previous
        elif mode == "delta":
            encoded = (values.astype(np.int32) - previous) & 0xffff
        else:
            encoded = values.astype(np.uint32)
        output.append(encoded)
    return output


def field_zigzag(fields):
    output = []
    for field in fields:
        values = field.astype(np.uint16)
        previous = np.empty_like(values)
        previous[0] = 0
        previous[1:] = values[:-1]
        delta = values.astype(np.int32) - previous.astype(np.int32)
        output.append(((delta << 1) ^ (delta >> 31)).astype(np.uint32))
    return output


def cross_field_xor(fields):
    output = []
    previous = {}
    for field in fields:
        values = field.astype(np.uint16)
        predictor_ = previous.get(len(values), np.zeros_like(values))
        output.append(values ^ predictor_)
        previous[len(values)] = values
    return output


def rans_nibbles(source, split):
    values = np.frombuffer(source, dtype=np.uint8)
    nibbles = np.empty(values.size * 2, dtype=np.uint8)
    nibbles[0::2] = values & 15
    nibbles[1::2] = values >> 4
    if not split:
        return len(rans_encode(nibbles, 16)) + 32
    total = 0
    for offset in range(4):
        total += len(rans_encode(nibbles[offset::4], 16)) + 32
    return total + 12


def context_rans_encode(codes, levels, context_shift, lanes=1):
    """Encode token-major symbols conditioned on the prior token's code."""
    values = np.asarray(codes, dtype=np.uint8)
    rows, width = values.shape
    contexts = np.zeros_like(values, dtype=np.uint8)
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
    payloads = []
    for lane in range(lanes):
        state = RANS_LOW
        emitted = []
        indices = range(lane, symbols.size, lanes)
        for index in reversed(indices):
            symbol = symbols[index]
            model = model_ids[index]
            frequency = int(frequencies[model][symbol])
            threshold = ((RANS_LOW >> RANS_BITS) << 8) * frequency
            while state >= threshold:
                emitted.append(state & 255)
                state >>= 8
            quotient, remainder = divmod(state, frequency)
            state = (quotient << RANS_BITS) + remainder
            state += int(cumulative[model][symbol])
        payloads.append(state.to_bytes(4, "little") + bytes(reversed(emitted)))

    tables = []
    for model in range(context_count):
        table = np.empty(RANS_TOTAL, dtype=np.int16)
        for symbol, frequency in enumerate(frequencies[model]):
            start = cumulative[model][symbol]
            table[start:start + frequency] = symbol
        tables.append(table)
    decoded = np.empty_like(symbols)
    states = [int.from_bytes(payload[:4], "little") for payload in payloads]
    positions = [4] * lanes
    for index in range(symbols.size):
        lane = index % lanes
        state = states[lane]
        payload = payloads[lane]
        row, column = divmod(index, width)
        previous = 0 if row == 0 else int(decoded[index - width])
        model = previous >> context_shift
        slot = state & (RANS_TOTAL - 1)
        symbol = int(tables[model][slot])
        decoded[index] = symbol
        state = int(frequencies[model][symbol]) * (state >> RANS_BITS)
        state += slot - int(cumulative[model][symbol])
        while state < RANS_LOW and positions[lane] < len(payload):
            state = (state << 8) | payload[positions[lane]]
            positions[lane] += 1
        states[lane] = state
    if not np.array_equal(decoded, symbols):
        raise RuntimeError("context rANS roundtrip failed")
    # The last frequency in each model follows from the fixed total.
    table_bytes = context_count * (levels - 1) * 2
    return b"".join(payloads), table_bytes


def uleb128(fields):
    output = bytearray()
    for field in fields:
        for original in field:
            value = int(original)
            while value >= 0x80:
                output.append((value & 0x7f) | 0x80)
                value >>= 7
            output.append(value)
    return bytes(output)


def decode_uleb128(source, lengths):
    output = []
    position = 0
    for length in lengths:
        field = np.empty(length, dtype=np.uint32)
        for index in range(length):
            value = 0
            shift = 0
            while True:
                byte = source[position]
                position += 1
                value |= (byte & 0x7f) << shift
                if byte < 0x80:
                    break
                shift += 7
            field[index] = value
        output.append(field)
    if position != len(source):
        raise RuntimeError("ULEB128 trailing data")
    return output


def undo_zigzag(fields):
    output = []
    for field in fields:
        previous = 0
        decoded = np.empty(len(field), dtype=np.uint16)
        for index, encoded in enumerate(field):
            encoded = int(encoded)
            delta = (encoded >> 1) ^ -(encoded & 1)
            previous += delta
            decoded[index] = previous
        output.append(decoded)
    return output


def metadata_bytes(fields):
    return b"".join(field.astype("<u2", copy=False).tobytes() for field in fields)


class NativeCompression:
    def __init__(self):
        self.library = ctypes.CDLL("/usr/lib/libcompression.dylib")
        self.library.compression_encode_buffer.restype = ctypes.c_size_t
        self.library.compression_encode_buffer.argtypes = (
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint32,
        )
        self.library.compression_decode_buffer.restype = ctypes.c_size_t
        self.library.compression_decode_buffer.argtypes = (
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint32,
        )

    def encode(self, source, algorithm):
        source = bytes(source)
        input_buffer = ctypes.create_string_buffer(source)
        output_buffer = ctypes.create_string_buffer(len(source) * 2 + 1024)
        size = self.library.compression_encode_buffer(
            output_buffer, len(output_buffer), input_buffer, len(source),
            None, algorithm,
        )
        if not size:
            raise RuntimeError("native compression failed")
        encoded = output_buffer.raw[:size]
        decoded = ctypes.create_string_buffer(len(source))
        decoded_size = self.library.compression_decode_buffer(
            decoded, len(source), encoded, len(encoded), None, algorithm,
        )
        if decoded_size != len(source) or decoded.raw != source:
            raise RuntimeError("native compression roundtrip failed")
        return encoded


def summarize(name, values):
    values = np.asarray(values, dtype=np.float64)
    print(
        f"{name:30s} mean={np.mean(values):.6f} "
        f"p95={np.percentile(values, 95):.6f} max={np.max(values):.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--sink", type=int, default=128)
    parser.add_argument("--all-results", action="store_true")
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    native = NativeCompression()
    rates = {}
    metadata_sizes = {}
    seen = set()
    dim = queries[0][6].size
    scalars = 2 * dim * args.group

    def add(name, byte_count):
        # One uint64 absolute tile offset provides independent random access.
        aligned = (byte_count + 8 + 15) // 16 * 16
        rates.setdefault(name, []).append(aligned * 8.0 / scalars)

    for layer, head, count, first, _, _, _, _ in queries:
        identity = layer, head, count, first
        if identity in seen:
            continue
        seen.add(identity)
        key_matrix = stack(keys, layer, head, count)[first:]
        value_matrix = stack(values, layer, head, count)[first:]
        for start in range(args.sink, count - args.group + 1, args.group):
            key_codes, key_fields = official_tile(
                key_matrix[start:start + args.group], 4, True
            )
            value_codes, value_fields = official_tile(
                value_matrix[start:start + args.group], 2, False
            )
            fields = key_fields + value_fields
            metadata = metadata_bytes(fields)
            raw_codes = pack_codes(key_codes, 4) + pack_codes(value_codes, 2)
            canonical = raw_codes + metadata

            key_payloads = {}
            value_payloads = {}
            for mode in ("raw", "delta", "xor"):
                for axis in ("flat", 0, 1):
                    label = f"{mode}-{axis}"
                    kp = predictor(key_codes, axis, mode)
                    vp = predictor(value_codes, axis, mode)
                    key_payloads[label] = rans_encode(kp, 16)
                    value_payloads[label] = rans_encode(vp, 4)

            # Two uint16 frequency tables and two payload lengths.
            code_overhead = 40 + 8
            for label in key_payloads:
                code_size = (
                    len(key_payloads[label]) + len(value_payloads[label])
                    + code_overhead
                )
                add(f"rans codes {label} + raw meta", code_size + len(metadata))

            metadata_variants = {"raw": metadata}
            for mode in ("raw", "xor", "delta"):
                encoded_fields = field_predict(fields, mode)
                metadata_variants[f"uleb-{mode}"] = uleb128(encoded_fields)
                transformed = b"".join(
                    field.astype("<u2", copy=False).tobytes()
                    for field in encoded_fields
                )
                metadata_variants[f"u16-{mode}"] = transformed
            zigzag = uleb128(field_zigzag(fields))
            restored = undo_zigzag(
                decode_uleb128(zigzag, [len(field) for field in fields])
            )
            if any(
                not np.array_equal(original, decoded)
                for original, decoded in zip(fields, restored)
            ):
                raise RuntimeError("metadata predictor roundtrip failed")
            metadata_variants["uleb-zigzag"] = zigzag
            metadata_variants["u16-cross-xor"] = metadata_bytes(
                cross_field_xor(fields)
            )

            for label, transformed in tuple(metadata_variants.items()):
                compressed = zlib.compress(transformed, 1)
                if zlib.decompress(compressed) != transformed:
                    raise RuntimeError("zlib roundtrip failed")
                metadata_variants[f"zlib-{label}"] = compressed
                metadata_variants[f"zlib9-{label}"] = zlib.compress(
                    transformed, 9
                )
                metadata_variants[f"bz2-{label}"] = bz2.compress(
                    transformed, 1
                )
                metadata_variants[f"lzma-{label}"] = lzma.compress(
                    transformed, preset=0
                )
                for native_name, algorithm in (
                    ("lz4", COMPRESSION_LZ4),
                    ("lzfse", COMPRESSION_LZFSE),
                ):
                    metadata_variants[f"{native_name}-{label}"] = (
                        native.encode(transformed, algorithm)
                    )

            raw_key = key_payloads["raw-flat"]
            raw_value = value_payloads["raw-flat"]
            code_size = len(raw_key) + len(raw_value) + code_overhead
            context_key, context_key_table = context_rans_encode(
                key_codes.T, 16, 2
            )
            context_value, context_value_table = context_rans_encode(
                value_codes, 4, 0
            )
            context_code_size = (
                len(context_key) + len(context_value)
                + context_key_table + context_value_table + 8
            )
            compact_key, compact_key_table = context_rans_encode(
                key_codes.T, 16, 3, lanes=4
            )
            compact_value, compact_value_table = context_rans_encode(
                value_codes, 4, 1, lanes=4
            )
            compact_code_size = (
                len(compact_key) + len(compact_value)
                + compact_key_table + compact_value_table + 16
            )
            q8_metadata_size = 3 * (dim + args.group) + 6 * 4
            add("rans codes + q8 field meta", code_size + q8_metadata_size)
            add(
                "four-state context-rans codes + q8 field meta",
                compact_code_size + q8_metadata_size,
            )
            for label, encoded in metadata_variants.items():
                metadata_sizes.setdefault(label, []).append(len(encoded))
                add(f"rans codes + {label} meta", code_size + len(encoded))
                add(
                    f"context-rans codes + {label} meta",
                    context_code_size + len(encoded),
                )
            for label in (
                "raw", "u16-xor", "u16-delta", "u16-cross-xor",
                "uleb-zigzag",
            ):
                transformed = metadata_variants[label]
                for split in (False, True):
                    nibble_size = rans_nibbles(transformed, split)
                    suffix = "split" if split else "single"
                    add(
                        f"rans codes + nibble-{suffix}-{label} meta",
                        code_size + nibble_size,
                    )

            for name, algorithm in (
                ("lz4 whole tile", COMPRESSION_LZ4),
                ("lzfse whole tile", COMPRESSION_LZFSE),
            ):
                add(name, len(native.encode(canonical, algorithm)) + 8)
            add("zlib whole tile", len(zlib.compress(canonical, 1)) + 8)

    print(
        f"tiles={len(next(iter(rates.values())))} dim={dim} group={args.group} "
        f"fp16_metadata_bytes={3 * (dim + args.group) * 2} "
        "rans_tables=40 header=8 absolute_offset=8 alignment=16"
    )
    metadata_items = sorted(
        metadata_sizes.items(), key=lambda item: np.mean(item[1])
    )
    for name, values_ in (
        metadata_items if args.all_results else metadata_items[:12]
    ):
        summarize(f"meta bytes {name}", values_)
    print()
    rate_items = sorted(rates.items(), key=lambda item: np.mean(item[1]))
    for name, values_ in (
        rate_items if args.all_results else rate_items[:30]
    ):
        summarize(name, values_)


if __name__ == "__main__":
    main()

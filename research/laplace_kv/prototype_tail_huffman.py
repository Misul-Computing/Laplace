#!/usr/bin/env python3
"""Measure a fixed canonical Huffman codec for K4/V2 tail records."""

import argparse
import math
import struct

import numpy as np

from prototype_delta import load_trace
from prototype_even_transform import factored_transform


# Symmetric, length-limited K4 table. The four central symbols have length 3;
# lengths grow with distance from the midpoint and stop at 6.
K_LENGTHS = (6, 6, 5, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 5, 6, 6)
# The positive central V2 symbol wins the deterministic equal-weight tie.
V_LENGTHS = (3, 2, 1, 3)


def canonical_codes(lengths):
    ordered = sorted(range(len(lengths)), key=lambda symbol: (lengths[symbol], symbol))
    codes = [None] * len(lengths)
    code = 0
    previous = 0
    for symbol in ordered:
        length = lengths[symbol]
        code <<= length - previous
        codes[symbol] = (code, length)
        code += 1
        previous = length
    if code != 1 << previous:
        raise ValueError("code lengths do not form a complete prefix code")
    return tuple(codes)


K_CODES = canonical_codes(K_LENGTHS)
V_CODES = canonical_codes(V_LENGTHS)


class BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.current = 0
        self.used = 0

    def write(self, value, bits):
        for shift in range(bits - 1, -1, -1):
            self.current = (self.current << 1) | ((value >> shift) & 1)
            self.used += 1
            if self.used == 8:
                self.data.append(self.current)
                self.current = 0
                self.used = 0

    def finish(self):
        if self.used:
            self.data.append(self.current << (8 - self.used))
            self.current = 0
            self.used = 0
        return bytes(self.data)


class BitReader:
    def __init__(self, data):
        self.data = data
        self.position = 0

    def read(self, bits):
        value = 0
        for _ in range(bits):
            if self.position >= len(self.data) * 8:
                raise ValueError("truncated bitstream")
            byte = self.data[self.position >> 3]
            shift = 7 - (self.position & 7)
            value = (value << 1) | ((byte >> shift) & 1)
            self.position += 1
        return value

    def check_padding(self):
        if len(self.data) * 8 - self.position >= 8:
            raise ValueError("unexpected trailing byte")
        while self.position < len(self.data) * 8:
            if self.read(1):
                raise ValueError("nonzero padding")


def decode_symbol(reader, codes):
    lookup = {(length, code): symbol for symbol, (code, length) in enumerate(codes)}
    code = 0
    for length in range(1, max(item[1] for item in codes) + 1):
        code = (code << 1) | reader.read(1)
        symbol = lookup.get((length, code))
        if symbol is not None:
            return symbol
    raise ValueError("invalid prefix code")


def quantize(vectors, bits):
    transformed = factored_transform(np.asarray(vectors, dtype=np.float32))
    levels = 1 << bits
    peak = levels / 2.0 - 0.5
    maximum = np.max(np.abs(transformed), axis=1)
    scale = np.where(maximum > 0.0, maximum / peak, 1.0)
    scale = scale.astype(np.float16)
    restored_scale = scale.astype(np.float32)
    if np.any(restored_scale == 0.0):
        raise ValueError("FP16 scale underflow")
    codes = np.clip(
        np.rint(transformed / restored_scale[:, None] + peak),
        0,
        levels - 1,
    ).astype(np.uint8)
    return scale, codes


def encode_entropy(key_scale, value_scale, key_codes, value_codes):
    writer = BitWriter()
    writer.write(0, 1)
    for symbol in key_codes:
        code, length = K_CODES[int(symbol)]
        writer.write(code, length)
    for symbol in value_codes:
        code, length = V_CODES[int(symbol)]
        writer.write(code, length)
    scales = struct.pack("<ee", float(key_scale), float(value_scale))
    return scales + writer.finish()


def encode_fallback(key_scale, value_scale, key_codes, value_codes):
    writer = BitWriter()
    writer.write(1, 1)
    for symbol in key_codes:
        writer.write(int(symbol), 3)
    for symbol in value_codes:
        writer.write(int(symbol), 2)
    scales = struct.pack("<ee", float(key_scale), float(value_scale))
    return scales + writer.finish()


def decode(record, dimension):
    key_scale, value_scale = struct.unpack("<ee", record[:4])
    reader = BitReader(record[4:])
    fallback = bool(reader.read(1))
    if fallback:
        keys = np.array([reader.read(3) for _ in range(dimension)], dtype=np.uint8)
        values = np.array([reader.read(2) for _ in range(dimension)], dtype=np.uint8)
    else:
        keys = np.array(
            [decode_symbol(reader, K_CODES) for _ in range(dimension)],
            dtype=np.uint8,
        )
        values = np.array(
            [decode_symbol(reader, V_CODES) for _ in range(dimension)],
            dtype=np.uint8,
        )
    reader.check_padding()
    return fallback, np.float16(key_scale), np.float16(value_scale), keys, values


def format_codes(codes):
    return " ".join(
        f"{symbol}:{code:0{length}b}"
        for symbol, (code, length) in enumerate(codes)
    )


def summarize(path):
    keys, values, _ = load_trace(path)
    identities = sorted(set(keys) & set(values))
    if len(identities) != len(keys) or len(identities) != len(values):
        raise ValueError("unpaired K/V trace records")
    key_vectors = np.stack([keys[identity] for identity in identities])
    value_vectors = np.stack([values[identity] for identity in identities])
    dimension = key_vectors.shape[1]
    key_scale4, key_codes4 = quantize(key_vectors, 4)
    value_scale2, value_codes2 = quantize(value_vectors, 2)
    key_scale3, key_codes3 = quantize(key_vectors, 3)

    cap = 6 * dimension // 8
    fallback_size = math.ceil((32 + 1 + 5 * dimension) / 8)
    sizes = []
    entropy_sizes = []
    fallback_positions = []
    for index, identity in enumerate(identities):
        entropy = encode_entropy(
            key_scale4[index], value_scale2[index],
            key_codes4[index], value_codes2[index],
        )
        entropy_sizes.append(len(entropy))
        if len(entropy) <= cap:
            record = entropy
            expected_key_scale = key_scale4[index]
            expected_key_codes = key_codes4[index]
            expected_fallback = False
        else:
            record = encode_fallback(
                key_scale3[index], value_scale2[index],
                key_codes3[index], value_codes2[index],
            )
            expected_key_scale = key_scale3[index]
            expected_key_codes = key_codes3[index]
            expected_fallback = True
            fallback_positions.append(identity)
        if len(record) > cap:
            raise RuntimeError("fallback exceeds the per-token cap")
        decoded = decode(record, dimension)
        if (
            decoded[0] != expected_fallback
            or decoded[1].view(np.uint16) != expected_key_scale.view(np.uint16)
            or decoded[2].view(np.uint16) != value_scale2[index].view(np.uint16)
            or not np.array_equal(decoded[3], expected_key_codes)
            or not np.array_equal(decoded[4], value_codes2[index])
        ):
            raise RuntimeError("tail record roundtrip failed")
        sizes.append(len(record))

    sizes = np.asarray(sizes)
    entropy_sizes = np.asarray(entropy_sizes)
    print(
        f"trace={path} vectors={len(identities)} dim={dimension} cap={cap} "
        f"fallback_bytes={fallback_size}"
    )
    print(
        f"entropy_candidate_bytes mean={np.mean(entropy_sizes):.6f} "
        f"p95={np.percentile(entropy_sizes, 95):.6f} "
        f"max={np.max(entropy_sizes)}"
    )
    print(
        f"stored_record_bytes mean={np.mean(sizes):.6f} "
        f"p95={np.percentile(sizes, 95):.6f} max={np.max(sizes)} "
        f"mean_bits_per_scalar={np.mean(sizes) * 8 / (2 * dimension):.6f}"
    )
    print(
        f"K3/V2_overflow={len(fallback_positions)}/{len(identities)} "
        f"({100 * len(fallback_positions) / len(identities):.6f}%)"
    )
    return identities, tuple(int(size) for size in sizes), fallback_positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+")
    args = parser.parse_args()
    print(f"K canonical: {format_codes(K_CODES)}")
    print(f"V canonical: {format_codes(V_CODES)}")
    results = [summarize(path) for path in args.traces]
    if len(results) == 2:
        if results[0] != results[1]:
            raise RuntimeError("the two trace record sets differ")
        print("trace_record_sets=identical")
    for dimension in (64, 96, 256):
        budget = 6 * dimension // 8
        fallback = math.ceil((5 * dimension + 33) / 8)
        entropy_limit = 6 * dimension - 33
        print(
            f"D{dimension}: cap={budget} fallback={fallback} "
            f"entropy_code_limit={entropy_limit} bits"
        )


if __name__ == "__main__":
    main()

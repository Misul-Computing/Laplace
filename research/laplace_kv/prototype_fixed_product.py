#!/usr/bin/env python3
"""Test fixed-width Gaussian product quantization on a captured KV trace."""

import argparse
import math
import struct

import numpy as np


HEADER = struct.Struct("<IIiiiiifi")
MAGIC = 0x4C4B5654


def load_trace(path):
    keys, values, queries = {}, {}, []
    with open(path, "rb") as trace:
        while header := trace.read(HEADER.size):
            if len(header) != HEADER.size:
                raise ValueError("truncated trace header")
            magic, kind, layer, head, position, dim, first, scale, index = HEADER.unpack(header)
            if magic != MAGIC or kind not in (1, 2, 3):
                raise ValueError("invalid trace record")
            count = dim * (2 if kind == 3 else 1)
            payload = np.fromfile(trace, dtype="<f4", count=count)
            if payload.size != count:
                raise ValueError("truncated trace payload")
            if kind == 1:
                keys[layer, head, position] = payload
            elif kind == 2:
                values[layer, head, position] = payload
            else:
                queries.append((layer, head, position, first, scale, index,
                                payload[:dim], payload[dim:]))
    return keys, values, queries


def fixed_gaussian_codebook(size, seed):
    """Build a deterministic, model-independent 2D normal codebook."""
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal((65536, 2)).astype(np.float32)
    codebook = samples[rng.choice(len(samples), size, replace=False)].copy()
    for _ in range(20):
        sums = np.zeros_like(codebook)
        counts = np.zeros(size, dtype=np.int64)
        for start in range(0, len(samples), 4096):
            batch = samples[start:start + 4096]
            nearest = np.argmin(
                np.sum((batch[:, None] - codebook[None]) ** 2, axis=2), axis=1
            )
            np.add.at(sums, nearest, batch)
            np.add.at(counts, nearest, 1)
        codebook = sums / np.maximum(counts[:, None], 1)
    table_scale = float(np.max(np.abs(codebook)) / 127.0)
    return np.rint(codebook / table_scale).astype(np.int8), table_scale


def reconstruct_table(codes, scale):
    return codes.astype(np.float32) * np.float32(scale)


def log8_scale(matrix):
    scale = np.sqrt(np.mean(matrix * matrix, axis=1, keepdims=True))
    field = np.clip(np.rint((np.log2(np.maximum(scale, 2.0 ** -8)) + 8) * 16), 0, 255)
    return np.exp2(field / 16 - 8)


def product_quantize(matrix, table):
    scale = log8_scale(matrix)
    pairs = (matrix / scale).reshape(-1, 2)
    decoded = np.empty_like(pairs)
    for start in range(0, len(pairs), 4096):
        batch = pairs[start:start + 4096]
        nearest = np.argmin(
            np.sum((batch[:, None] - table[None]) ** 2, axis=2), axis=1
        )
        decoded[start:start + len(batch)] = table[nearest]
    return decoded.reshape(matrix.shape) * scale


def stack(records, layer, head, count):
    return np.stack([records[layer, head, token] for token in range(count)]).astype(np.float64)


def softmax(scores):
    weights = np.exp(scores - np.max(scores))
    return weights / np.sum(weights)


def relative_error(candidate, reference):
    return np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), 1e-20)


def summary(values):
    values = np.asarray(values) * 100
    return f"mean={np.mean(values):.3f}% p95={np.percentile(values, 95):.3f}% max={np.max(values):.3f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    if not queries or queries[0][6].size % 2:
        raise ValueError("trace must contain even-width vectors")

    key_codes, key_scale = fixed_gaussian_codebook(128, 417)
    value_codes, value_scale = fixed_gaussian_codebook(16, 418)
    key_table = reconstruct_table(key_codes, key_scale)
    value_table = reconstruct_table(value_codes, value_scale)

    errors = {"K-only": [], "V-only": [], "K+V": []}
    cache = {}
    for layer, head, count, first, scale, _, query, _ in queries:
        group = layer, head, count, first
        if group not in cache:
            original_k = stack(keys, layer, head, count)[first:]
            original_v = stack(values, layer, head, count)[first:]
            cache[group] = (
                original_k,
                original_v,
                product_quantize(original_k, key_table),
                product_quantize(original_v, value_table),
            )
        original_k, original_v, candidate_k, candidate_v = cache[group]
        query = query.astype(np.float64)
        exact_weights = softmax(np.einsum("td,d->t", original_k, query) * scale)
        candidate_weights = softmax(np.einsum("td,d->t", candidate_k, query) * scale)
        exact = np.einsum("t,td->d", exact_weights, original_v)
        errors["K-only"].append(relative_error(
            np.einsum("t,td->d", candidate_weights, original_v), exact
        ))
        errors["V-only"].append(relative_error(
            np.einsum("t,td->d", exact_weights, candidate_v), exact
        ))
        errors["K+V"].append(relative_error(
            np.einsum("t,td->d", candidate_weights, candidate_v), exact
        ))

    dim = queries[0][6].size
    unique_rows = len(keys)
    scalar_count = unique_rows * dim * 2
    payload_bits = unique_rows * (dim * 7 // 2 + dim * 4 // 2 + 16)
    table_bytes = math.ceil((key_codes.nbytes + 4) / 16) * 16
    table_bytes += math.ceil((value_codes.nbytes + 4) / 16) * 16
    effective_bits = (payload_bits + table_bytes * 8) / scalar_count
    one_head_tile_bits = (64 * (dim * 7 // 2 + dim * 4 // 2 + 16)
                          + table_bytes * 8) / (64 * dim * 2)
    print(
        f"format=K2D-7b/V2D-4b scale=log8 tables=int8 "
        f"payload={payload_bits / scalar_count:.6f} bits/scalar"
    )
    print(
        f"table={table_bytes} bytes fully_counted={effective_bits:.6f} bits/scalar "
        f"one_head_64_token={one_head_tile_bits:.6f} bits/scalar"
    )
    for name in ("K-only", "V-only", "K+V"):
        print(f"{name:6s} {summary(errors[name])}")


if __name__ == "__main__":
    main()

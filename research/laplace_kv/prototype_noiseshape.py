#!/usr/bin/env python3
"""Test zero-overhead error-shaped rounding on normalized KV tiles."""

import argparse

import numpy as np

from prototype_delta import (
    hadamard_rows,
    load_trace,
    relative_error,
    softmax,
    stack,
    summarize,
    variance_normalize,
)


def quantize_tile(source, bits, key, shaped):
    rotated = hadamard_rows(source)
    oriented = rotated.T if key else rotated
    balanced, columns, rows = variance_normalize(oriented)
    low = np.min(balanced, axis=1, keepdims=True)
    high = np.max(balanced, axis=1, keepdims=True)
    maximum = (1 << bits) - 1
    step = np.maximum((high - low) / maximum, 1e-10)
    if not shaped:
        codes = np.clip(np.rint((balanced - low) / step), 0, maximum)
    else:
        codes = np.empty_like(balanced)
        error = np.zeros(oriented.shape[1], dtype=np.float64)
        for row in range(oriented.shape[0]):
            target = oriented[row] - error
            denominator = rows[row, 0] * columns[0]
            wanted = target / denominator
            codes[row] = np.clip(
                np.rint((wanted - low[row, 0]) / step[row, 0]),
                0, maximum,
            )
            restored = (
                codes[row] * step[row, 0] + low[row, 0]
            ) * denominator
            error += restored - oriented[row]
    restored = (codes * step + low) * columns * rows
    restored = restored.T if key else restored
    return hadamard_rows(restored)


def quantize(matrix, bits, key, group, shaped):
    output = np.empty_like(matrix)
    for start in range(0, matrix.shape[0], group):
        source = matrix[start:start + group]
        output[start:start + source.shape[0]] = quantize_tile(
            source, bits, key, shaped
        )
    return output


def effective_bits(key_bits, value_bits, dim, group):
    return (key_bits + value_bits) / 2.0 + 24.0 * (dim + group) / (dim * group)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--group", type=int, default=256)
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    modes = ((4, 2, False), (4, 2, True), (4, 1, False), (4, 1, True),
             (3, 1, False), (3, 1, True), (2, 2, True))
    errors = {mode: [] for mode in modes}
    cache = {}

    for layer, head, count, first, scale, _, query, _ in queries:
        key = layer, head, count, first
        if key not in cache:
            original_k = stack(keys, layer, head, count)[first:]
            original_v = stack(values, layer, head, count)[first:]
            variants = {}
            for key_bits, value_bits, shaped in modes:
                variants[key_bits, value_bits, shaped] = (
                    quantize(original_k, key_bits, True, args.group, False),
                    quantize(original_v, value_bits, False, args.group, shaped),
                )
            cache[key] = original_k, original_v, variants
        original_k, original_v, variants = cache[key]
        exact_weights = softmax(np.einsum("td,d->t", original_k, query) * scale)
        exact = np.einsum("t,td->d", exact_weights, original_v)
        for mode, result in errors.items():
            candidate_k, candidate_v = variants[mode]
            weights = softmax(np.einsum("td,d->t", candidate_k, query) * scale)
            candidate = np.einsum("t,td->d", weights, candidate_v)
            result.append(relative_error(candidate, exact))

    dim = queries[0][6].size
    print(f"traces={len(queries)} group={args.group} dim={dim} metadata=fp16")
    for (key_bits, value_bits, shaped), result in errors.items():
        bits = effective_bits(key_bits, value_bits, dim, args.group)
        suffix = "shaped" if shaped else "rtn"
        summarize(f"K{key_bits}/V{value_bits}/{suffix}/{bits:.3f}b", result)


if __name__ == "__main__":
    main()

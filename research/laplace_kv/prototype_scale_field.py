#!/usr/bin/env python3
"""Evaluate the preregistered compact Laplace scale-field candidate."""

import argparse
import math

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack, summarize
from prototype_kvarn_official import quantize_tile


def quantize(matrix, bits, key, group, sink, metadata_bits):
    output = matrix.copy()
    for start in range(sink, matrix.shape[0] - group + 1, group):
        output[start:start + group] = quantize_tile(
            matrix[start:start + group], bits, key, metadata_bits
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--group", type=int, default=256)
    parser.add_argument("--sink", type=int, default=0)
    parser.add_argument("--metadata-bits", type=int, default=8)
    parser.add_argument("--key-bits", type=int, default=3)
    parser.add_argument("--value-bits", type=int, default=2)
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    modes = (0, args.metadata_bits, "fp8")
    errors = {mode: [] for mode in modes}
    cache = {}
    for layer, head, count, first, scale, _, query, _ in queries:
        cache_key = layer, head, count, first
        if cache_key not in cache:
            original_k = stack(keys, layer, head, count)[first:]
            original_v = stack(values, layer, head, count)[first:]
            variants = {
                metadata_bits: (
                    quantize(
                        original_k, args.key_bits, True,
                        args.group, args.sink, metadata_bits
                    ),
                    quantize(
                        original_v, args.value_bits, False,
                        args.group, args.sink, metadata_bits
                    ),
                )
                for metadata_bits in errors
            }
            cache[cache_key] = original_k, original_v, variants
        original_k, original_v, variants = cache[cache_key]
        exact_weights = softmax(np.einsum("td,d->t", original_k, query) * scale)
        exact = np.einsum("t,td->d", exact_weights, original_v)
        for metadata_bits, result in errors.items():
            candidate_k, candidate_v = variants[metadata_bits]
            weights = softmax(np.einsum("td,d->t", candidate_k, query) * scale)
            candidate = np.einsum("t,td->d", weights, candidate_v)
            result.append(relative_error(candidate, exact))

    dim = queries[0][6].size
    metadata_bytes = math.ceil(
        3 * (dim + args.group) * args.metadata_bits / 8
    ) + 24
    tile_bits = (args.key_bits + args.value_bits) / 2.0
    tile_bits += metadata_bytes * 8.0 / (2 * dim * args.group)
    print(
        f"traces={len(queries)} group={args.group} sink={args.sink} "
        f"tile_bits={tile_bits:.6f}"
    )
    summarize("fp16 metadata", errors[0])
    summarize(f"q{args.metadata_bits} scale field", errors[args.metadata_bits])
    summarize("polarity fp8", errors["fp8"])


if __name__ == "__main__":
    main()

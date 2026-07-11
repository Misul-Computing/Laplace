#!/usr/bin/env python3
"""Test fixed-width KV quantization using softmax key-offset invariance."""

import argparse

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack, summarize
from prototype_kvarn_official import (
    even_transform_rows,
    scale_field,
    store_even_transform_rows,
)


def quantize_key(source, metadata_bits):
    rotated = store_even_transform_rows(source)
    oriented = rotated.T
    low = np.min(oriented, axis=1, keepdims=True)
    high = np.max(oriented, axis=1, keepdims=True)
    bits = 3 + (np.arange(oriented.shape[0])[:, None] % 4 != 0)
    maximum = (1 << bits) - 1
    step = np.maximum((high - low) / maximum, 1e-10)
    codes = np.clip(np.rint((oriented - low) / step), 0, maximum)
    stored_step = scale_field(step.ravel(), metadata_bits)[:, None]
    # The omitted low contributes one query-dependent constant to every logit.
    restored = (codes * stored_step).T
    return even_transform_rows(restored).astype(np.float32)


def quantize_value(source, metadata_bits):
    rotated = store_even_transform_rows(source)
    oriented = rotated.T
    low = np.min(oriented, axis=1, keepdims=True)
    high = np.max(oriented, axis=1, keepdims=True)
    step = np.maximum((high - low) / 3.0, 1e-10)
    codes = np.clip(np.rint((oriented - low) / step), 0, 3)
    stored_step = scale_field(step.ravel(), metadata_bits)[:, None]
    stored_low = scale_field(low.ravel(), metadata_bits)[:, None]
    restored = (codes * stored_step + stored_low).T
    return even_transform_rows(restored).astype(np.float32)


def quantize(matrix, key, group, sink, metadata_bits):
    output = matrix.copy()
    for start in range(sink, matrix.shape[0] - group + 1, group):
        tile = matrix[start:start + group]
        output[start:start + group] = (
            quantize_key(tile, metadata_bits)
            if key else quantize_value(tile, metadata_bits)
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--sink", type=int, default=128)
    parser.add_argument("--metadata-bits", type=int, default=8)
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    errors = []
    cache = {}
    for layer, head, count, first, scale, _, query, _ in queries:
        identity = layer, head, count, first
        if identity not in cache:
            original_k = stack(keys, layer, head, count)[first:]
            original_v = stack(values, layer, head, count)[first:]
            cache[identity] = (
                original_k,
                original_v,
                quantize(original_k, True, args.group, args.sink,
                         args.metadata_bits),
                quantize(original_v, False, args.group, args.sink,
                         args.metadata_bits),
            )
        original_k, original_v, candidate_k, candidate_v = cache[identity]
        exact_weights = softmax(np.einsum("td,d->t", original_k, query) * scale)
        exact = np.einsum("t,td->d", exact_weights, original_v)
        weights = softmax(np.einsum("td,d->t", candidate_k, query) * scale)
        candidate = np.einsum("t,td->d", weights, candidate_v)
        errors.append(relative_error(candidate, exact))

    dim = queries[0][6].size
    metadata_bytes = 3 * dim + 12
    code_rate = ((3.75 + 2.0) / 2.0)
    bits = code_rate + metadata_bytes * 8.0 / (2 * dim * args.group)
    print(
        f"traces={len(queries)} group={args.group} sink={args.sink} "
        f"metadata_bytes={metadata_bytes} tile_bits={bits:.6f}"
    )
    summarize("affine K3.75/V2", errors)


if __name__ == "__main__":
    main()

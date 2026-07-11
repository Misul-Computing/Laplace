#!/usr/bin/env python3
"""Test optimistic sparse and low-rank value representations with exact keys."""

import argparse

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack, summarize


def sparse_rows(values, fraction):
    retained = max(1, int(values.shape[1] * fraction))
    output = np.zeros_like(values)
    indices = np.argpartition(np.abs(values), -retained, axis=1)[:, -retained:]
    rows = np.arange(values.shape[0])[:, None]
    output[rows, indices] = values[rows, indices]
    return output


def low_rank(values, rank):
    left, singular, right = np.linalg.svd(values.astype(np.float64), full_matrices=False)
    return np.einsum(
        "tr,rd->td", left[:, :rank] * singular[:rank], right[:rank]
    )


def tiled_low_rank(values, rank, tile=64):
    output = np.empty_like(values, dtype=np.float64)
    for start in range(0, values.shape[0], tile):
        block = values[start:start + tile]
        output[start:start + block.shape[0]] = low_rank(block, rank)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    fractions = (0.0625, 0.125, 0.1875, 0.25)
    full_ranks = (4, 6, 8)
    tile_ranks = (2, 4, 6)
    errors = {("sparse", value): [] for value in fractions}
    errors.update({("full-rank", value): [] for value in full_ranks})
    errors.update({("tile-rank", value): [] for value in tile_ranks})
    cache = {}

    for layer, head, count, first, scale, _, query, _ in queries:
        key = layer, head, count, first
        if key not in cache:
            k = stack(keys, layer, head, count)[first:]
            v = stack(values, layer, head, count)[first:]
            variants = {}
            for fraction in fractions:
                variants["sparse", fraction] = sparse_rows(v, fraction)
            for rank in full_ranks:
                variants["full-rank", rank] = low_rank(v, rank)
            for rank in tile_ranks:
                variants["tile-rank", rank] = tiled_low_rank(v, rank)
            cache[key] = k, v, variants
        k, v, variants = cache[key]
        weights = softmax(np.einsum("td,d->t", k, query) * scale)
        exact = np.einsum("t,td->d", weights, v)
        for variant, result in errors.items():
            candidate = np.einsum("t,td->d", weights, variants[variant])
            result.append(relative_error(candidate, exact))

    tokens = queries[0][2] - queries[0][3]
    dim = queries[0][6].size
    print(f"traces={len(queries)} exact_keys fp64_oracle metadata=free")
    for (mode, value), result in errors.items():
        if mode == "sparse":
            bits = 16.0 * value
        elif mode == "full-rank":
            bits = 16.0 * value * (tokens + dim) / (tokens * dim)
        else:
            bits = 16.0 * value * (64 + dim) / (64 * dim)
        summarize(f"{mode}{value}/{bits:.2f}b", result)


if __name__ == "__main__":
    main()

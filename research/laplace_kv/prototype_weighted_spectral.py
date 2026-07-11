#!/usr/bin/env python3
"""Screen attention-weighted low-rank correction on captured KV traces."""

import argparse
from collections import defaultdict

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack
from prototype_eoptshrinkq import (
    attention,
    gaussian_lloyd,
    half,
    matmul,
    pair_layout,
    quantized_factors,
    summarize,
    tq_mse,
)
from prototype_syndrome_embedding import encode_pair


TILE = 128
KEY_RANK = 7
VALUE_RANK = 5


def hadamard(dimension):
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < dimension:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    if matrix.shape != (dimension, dimension):
        raise ValueError("head dimension must be a power of two")
    return matrix / np.sqrt(dimension)


def rank_approximation(matrix, rank):
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    return matmul(left[:, :rank] * singular[:rank], right[:rank])


def pseudo_inverse(matrix):
    with np.errstate(all="ignore"):
        inverse = np.linalg.pinv(matrix, rcond=1e-6)
    if not np.all(np.isfinite(inverse)):
        raise FloatingPointError("non-finite pseudoinverse")
    return inverse


def right_weighted_correction(error, probes, rank):
    measured = matmul(error, probes.T)
    return matmul(
        rank_approximation(measured, rank),
        pseudo_inverse(probes.T),
    )


def left_weighted_correction(error, probes, rank):
    measured = matmul(probes, error)
    return matmul(
        pseudo_inverse(probes),
        rank_approximation(measured, rank),
    )


def quantized_correction(correction, rank):
    left, singular, right = np.linalg.svd(correction, full_matrices=False)
    return quantized_factors(
        left[:, :rank], singular[:rank], right[:rank]
    )


def base_tile(matrix, rotation, centers):
    decoded, _, _ = tq_mse(matrix, rotation, centers)
    return decoded


def grouped_queries(queries):
    grouped = defaultdict(list)
    for record in queries:
        grouped[record[0], record[1]].append(record)
    for records in grouped.values():
        records.sort(key=lambda item: (item[2], item[5]))
    return grouped


def current_query_screen(keys, values, grouped, rotation, centers, start):
    names = (
        "K4/V2-Q8",
        "residual-SVD-Q4",
        "attention-oracle-Q4",
        "attention-oracle-exact",
    )
    errors = {name: [] for name in names}
    score_errors = {name: [] for name in names}
    tiles = 0

    for (layer, head), history in grouped.items():
        count = max(record[2] for record in history)
        records = [record for record in history if record[2] == count]
        first = records[0][3]
        original_k = half(stack(keys, layer, head, count)[first:])
        original_v = half(stack(values, layer, head, count)[first:])
        queries = np.stack([record[6] for record in records]).astype(np.float64)
        scales = np.asarray([record[4] for record in records])
        scores = matmul(original_k, queries.T) * scales
        weights = np.exp(scores - np.max(scores, axis=0))
        weights /= np.sum(weights, axis=0)
        variants = {
            name: [original_k.copy(), original_v.copy()] for name in names
        }

        for offset in range(start, len(original_k), TILE):
            region = slice(offset, offset + TILE)
            if len(original_k[region]) != TILE:
                raise ValueError("screen requires complete 128-token tiles")
            base_k = base_tile(original_k[region], rotation, centers)
            base_v = base_tile(original_v[region], rotation, centers)
            error_k = original_k[region] - base_k
            error_v = original_v[region] - base_v
            correction_k = right_weighted_correction(
                error_k, queries, KEY_RANK
            )
            correction_v = left_weighted_correction(
                error_v, weights[region].T, VALUE_RANK
            )
            svd_k = rank_approximation(error_k, KEY_RANK)
            svd_v = rank_approximation(error_v, VALUE_RANK)
            reference_k, reference_v, _, _ = encode_pair(
                original_k[region], original_v[region], 8
            )
            variants["K4/V2-Q8"][0][region] = reference_k
            variants["K4/V2-Q8"][1][region] = reference_v
            variants["residual-SVD-Q4"][0][region] = (
                base_k + quantized_correction(svd_k, KEY_RANK)
            )
            variants["residual-SVD-Q4"][1][region] = (
                base_v + quantized_correction(svd_v, VALUE_RANK)
            )
            variants["attention-oracle-Q4"][0][region] = (
                base_k + quantized_correction(correction_k, KEY_RANK)
            )
            variants["attention-oracle-Q4"][1][region] = (
                base_v + quantized_correction(correction_v, VALUE_RANK)
            )
            variants["attention-oracle-exact"][0][region] = base_k + correction_k
            variants["attention-oracle-exact"][1][region] = base_v + correction_v
            tiles += 1

        for record in records:
            query = record[6]
            scale = record[4]
            exact_scores = matmul(original_k, query) * scale
            exact = attention(query, scale, original_k, original_v)
            for name, (candidate_k, candidate_v) in variants.items():
                candidate_scores = matmul(candidate_k, query) * scale
                score_errors[name].append(
                    relative_error(candidate_scores, exact_scores)
                )
                errors[name].append(relative_error(
                    attention(query, scale, candidate_k, candidate_v), exact
                ))
    return errors, score_errors, tiles


def causal_screen(keys, values, grouped, rotation, centers):
    names = ("K4/V2-Q8", "causal-Q4", "causal-exact", "residual-SVD-Q4")
    errors = {name: [] for name in names}
    score_errors = {name: [] for name in names}

    for (layer, head), records in grouped.items():
        count = max(record[2] for record in records)
        if count < 2 * TILE:
            continue
        original_k = half(stack(keys, layer, head, count))
        original_v = half(stack(values, layer, head, count))
        base_k = base_tile(original_k[:TILE], rotation, centers)
        base_v = base_tile(original_v[:TILE], rotation, centers)
        error_k = original_k[:TILE] - base_k
        error_v = original_v[:TILE] - base_v
        history = [record for record in records if record[2] <= TILE]
        queries = np.stack([record[6] for record in history]).astype(np.float64)
        weights = []
        for record in history:
            observed = record[2]
            row = np.zeros(TILE, dtype=np.float64)
            row[:observed] = softmax(
                matmul(original_k[:observed], record[6]) * record[4]
            )
            weights.append(row)
        correction_k = right_weighted_correction(error_k, queries, KEY_RANK)
        correction_v = left_weighted_correction(
            error_v, np.stack(weights), VALUE_RANK
        )
        svd_k = rank_approximation(error_k, KEY_RANK)
        svd_v = rank_approximation(error_v, VALUE_RANK)
        reference_k, reference_v, _, _ = encode_pair(
            original_k[:TILE], original_v[:TILE], 8
        )
        variants = {
            "K4/V2-Q8": (reference_k, reference_v),
            "causal-Q4": (
                base_k + quantized_correction(correction_k, KEY_RANK),
                base_v + quantized_correction(correction_v, VALUE_RANK),
            ),
            "causal-exact": (base_k + correction_k, base_v + correction_v),
            "residual-SVD-Q4": (
                base_k + quantized_correction(svd_k, KEY_RANK),
                base_v + quantized_correction(svd_v, VALUE_RANK),
            ),
        }

        for record in records:
            observed = record[2]
            if observed <= TILE:
                continue
            query = record[6]
            scale = record[4]
            exact_scores = matmul(original_k[:observed], query) * scale
            exact_weights = softmax(exact_scores)
            exact = matmul(exact_weights, original_v[:observed])
            for name, (tile_k, tile_v) in variants.items():
                candidate_scores = exact_scores.copy()
                candidate_scores[:TILE] = matmul(tile_k, query) * scale
                candidate_weights = softmax(candidate_scores)
                candidate = (
                    matmul(candidate_weights[:TILE], tile_v)
                    + matmul(
                        candidate_weights[TILE:], original_v[TILE:observed]
                    )
                )
                score_errors[name].append(
                    relative_error(candidate_scores, exact_scores)
                )
                errors[name].append(relative_error(candidate, exact))
    return errors, score_errors


def print_results(errors, score_errors):
    for name in errors:
        print(f"{name:25s} attention {summarize(errors[name])}")
        print(f"{'':25s} K-score  {summarize(score_errors[name])}")


def print_storage(tile_pairs):
    layout = pair_layout(TILE, 64, KEY_RANK, VALUE_RANK)
    table_bytes = 4 * 4
    table_rate = table_bytes * 8 / (tile_pairs * 2 * TILE * 64)
    print(
        f"record={layout['addressed_bytes']} bytes "
        f"counted={layout['counted']:.6f} bits/scalar "
        f"implicit-WH+FP32-codebook={layout['counted'] + table_rate:.6f}"
    )
    for context in (4095, 16383, 65535):
        sealed, tail = divmod(context, TILE)
        total = (
            sealed * layout["addressed_bytes"]
            + tail * 2 * 64 * 2
            + table_bytes
        )
        rate = total * 8 / (2 * context * 64)
        print(f"context={context} tail={tail} effective={rate:.6f} bits/scalar")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="all-token D64 capture")
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    grouped = grouped_queries(queries)
    dimension = queries[0][6].size
    counts = {record[2] for record in queries}
    if dimension != 64 or len(counts) < 2 * TILE:
        raise ValueError("requires an all-token D64 trace through token 256")
    rotation = hadamard(dimension)
    centers = gaussian_lloyd(2, dimension)

    for label, start in (("sealed-tile", TILE), ("full-prefix", 0)):
        print(f"\n[{label}, invalid current-query oracle]")
        errors, score_errors, tiles = current_query_screen(
            keys, values, grouped, rotation, centers, start
        )
        print_results(errors, score_errors)
        print_storage(tiles)

    print("\n[causal first-tile metric, train<=128 test=129..256]")
    errors, score_errors = causal_screen(
        keys, values, grouped, rotation, centers
    )
    print_results(errors, score_errors)


if __name__ == "__main__":
    main()

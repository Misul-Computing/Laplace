#!/usr/bin/env python3
"""Screen causal backward value-forest residual coding."""

import argparse
import math

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack
from prototype_kvarn_official import (
    even_transform_rows,
    half,
    polarity_fp8,
    store_even_transform_rows,
)
from prototype_mixed_radix import quantize_tile


TILE = 128
WINDOW = 15


def asymmetric_k4_rows(source):
    transformed = store_even_transform_rows(source).astype(np.float64)
    low = np.min(transformed, axis=1, keepdims=True)
    high = np.max(transformed, axis=1, keepdims=True)
    step = np.maximum((high - low) / 15.0, 1e-10)
    zero = np.clip(np.rint(-low / step), 0, 15)
    stored_step = np.maximum(half(step), 2.0 ** -24)
    codes = np.clip(np.rint(transformed / step + zero), 0, 15)
    decoded = (codes - zero) * stored_step
    return half(even_transform_rows(decoded)).astype(np.float64)


def sign_residual(residual):
    scale = float(polarity_fp8([np.mean(np.abs(residual))], False)[0])
    if scale == 0:
        return np.zeros_like(residual)
    return np.where(residual < 0, -scale, scale)


def value_forest(source, window):
    transformed = store_even_transform_rows(source).astype(np.float64)
    decoded = np.empty_like(transformed)
    residuals = np.empty_like(transformed)
    parents = np.full(len(source), -1, dtype=np.int16)
    zero = np.zeros(source.shape[1], dtype=np.float64)
    for token, row in enumerate(transformed):
        candidates = [(-1, zero)]
        for parent in range(max(0, token - window), token):
            candidates.append((parent, decoded[parent]))
        best = None
        for parent, anchor in candidates:
            residual = sign_residual(row - anchor)
            reconstruction = anchor + residual
            error = np.sum((row - reconstruction) ** 2)
            if best is None or error < best[0]:
                best = error, parent, residual, reconstruction
        _, parents[token], residuals[token], decoded[token] = best
    restored = half(even_transform_rows(decoded)).astype(np.float64)
    return restored, parents, residuals


def forest_attention(weights, prefix, parents, residuals):
    split = len(prefix)
    output = np.einsum("t,td->d", weights[:split], prefix)
    masses = weights[split:split + len(parents)].astype(np.float64).copy()
    transformed = np.zeros(residuals.shape[1], dtype=np.float64)
    for token in range(len(parents) - 1, -1, -1):
        transformed += masses[token] * residuals[token]
        if parents[token] >= 0:
            masses[parents[token]] += masses[token]
    return output + even_transform_rows(transformed[None, :])[0]


def attention(keys, values, query, scale):
    weights = softmax(np.einsum("td,d->t", keys, query) * scale)
    return weights, np.einsum("t,td->d", weights, values)


def summarize(label, values):
    values = np.asarray(values) * 100.0
    print(
        f"{label:22s} mean={np.mean(values):8.3f}% "
        f"p95={np.percentile(values, 95):8.3f}% "
        f"max={np.max(values):8.3f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    context = max(item[2] - item[3] for item in queries)
    queries = [item for item in queries if item[2] - item[3] == context]
    dimension = queries[0][6].size
    if context < 2 * TILE:
        raise ValueError("trace needs at least 256 tokens")

    logical_rate = 2.5 + 14.0 / dimension
    raw_bytes = ((5 * dimension + 28) * TILE + 7) // 8 + 16
    record_bytes = math.ceil(raw_bytes / 64) * 64
    complete_rate = record_bytes * 8.0 / (2 * dimension * TILE)
    labels = (
        "K4/V2 Q8",
        "exact-K/zero-R1",
        "exact-K/forest-R1",
        "row-K4/exact-V",
        "row-K4/zero-R1",
        "row-K4/forest-R1",
    )
    errors = {label: [] for label in labels}
    cache = {}
    region = slice(TILE, 2 * TILE)
    identity_error = 0.0

    for layer, head, count, first, scale, _, query, _ in queries:
        identity = layer, head, count, first
        if identity not in cache:
            original_k = half(stack(keys, layer, head, count)[first:])
            original_v = half(stack(values, layer, head, count)[first:])
            control_k = original_k.copy()
            control_v = original_v.copy()
            control_k[region], _ = quantize_tile(
                original_k[region], 16, True, 8
            )
            control_v[region], _ = quantize_tile(
                original_v[region], 4, False, 8
            )
            row_k4 = original_k.copy()
            row_k4[region] = asymmetric_k4_rows(original_k[region])
            zero_v, zero_parents, zero_residuals = value_forest(
                original_v[region], 0
            )
            forest_v, forest_parents, forest_residuals = value_forest(
                original_v[region], WINDOW
            )
            candidate_zero_v = original_v.copy()
            candidate_forest_v = original_v.copy()
            candidate_zero_v[region] = zero_v
            candidate_forest_v[region] = forest_v
            cache[identity] = (
                original_k,
                original_v,
                control_k,
                control_v,
                row_k4,
                candidate_zero_v,
                candidate_forest_v,
                zero_parents,
                zero_residuals,
                forest_parents,
                forest_residuals,
            )

        (original_k, original_v, control_k, control_v, row_k4,
         candidate_zero_v, candidate_forest_v, zero_parents, zero_residuals,
         forest_parents, forest_residuals) = cache[identity]
        _, reference = attention(original_k, original_v, query, scale)
        variants = {
            "K4/V2 Q8": (control_k, control_v, None),
            "exact-K/zero-R1": (
                original_k, candidate_zero_v, (zero_parents, zero_residuals)
            ),
            "exact-K/forest-R1": (
                original_k, candidate_forest_v,
                (forest_parents, forest_residuals),
            ),
            "row-K4/exact-V": (row_k4, original_v, None),
            "row-K4/zero-R1": (
                row_k4, candidate_zero_v, (zero_parents, zero_residuals)
            ),
            "row-K4/forest-R1": (
                row_k4, candidate_forest_v,
                (forest_parents, forest_residuals),
            ),
        }
        for label, (candidate_k, candidate_v, forest) in variants.items():
            weights, output = attention(candidate_k, candidate_v, query, scale)
            if forest:
                direct = forest_attention(
                    weights, original_v[:TILE], forest[0], forest[1]
                )
                identity_error = max(
                    identity_error, relative_error(direct, output)
                )
            errors[label].append(relative_error(output, reference))

    print(
        f"trace={args.trace} queries={len(queries)} D={dimension} "
        f"window={WINDOW} logical_rate={logical_rate:.6f} "
        f"complete_rate={complete_rate:.6f} bits/scalar"
    )
    for label in labels:
        summarize(label, errors[label])
    print(f"forest direct-identity max={identity_error:.3e}")
    if identity_error > 5e-4:
        raise RuntimeError("forest direct attention identity failed")


if __name__ == "__main__":
    main()

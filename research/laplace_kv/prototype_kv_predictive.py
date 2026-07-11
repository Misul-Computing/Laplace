#!/usr/bin/env python3
"""Screen causal same-token K-to-V predictive residual coding."""

import argparse

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack
from prototype_kvarn_official import (
    even_transform_rows,
    half,
    store_even_transform_rows,
)
from prototype_syndrome_embedding import encode_pair


TRAIN = 128


def midrise_rows(matrix, bits):
    transformed = store_even_transform_rows(matrix).astype(np.float64)
    levels = 1 << bits
    peak = levels / 2.0 - 0.5
    maximum = np.max(np.abs(transformed), axis=1, keepdims=True)
    scale = np.maximum(
        half(np.where(maximum > 0, maximum / peak, 1.0)), 2.0 ** -24
    )
    codes = np.clip(
        np.rint(transformed / scale + peak), 0, levels - 1
    )
    decoded = (codes - peak) * scale
    return half(even_transform_rows(decoded)).astype(np.float64)


def sign_rows(matrix):
    transformed = store_even_transform_rows(matrix).astype(np.float64)
    scale = half(np.mean(np.abs(transformed), axis=1, keepdims=True))
    decoded = np.where(transformed < 0, -scale, scale)
    return half(even_transform_rows(decoded)).astype(np.float64)


def fit_map(keys, values):
    source = keys[:TRAIN].astype(np.float64)
    target = values[:TRAIN].astype(np.float64)
    gram = np.einsum("ti,tj->ij", source, source)
    ridge = 1e-3 * np.trace(gram) / max(1, gram.shape[0])
    return np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0]),
        np.einsum("ti,tj->ij", source, target),
    )


def attention(keys, values, query, scale):
    weights = softmax(np.einsum("td,d->t", keys, query) * scale)
    return np.einsum("t,td->d", weights, values)


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
    if context < 2 * TRAIN:
        raise ValueError("trace needs at least 256 tokens")

    body_rate = (5 * dimension + 32) / (2 * dimension)
    print(
        f"trace={args.trace} queries={len(queries)} D={dimension} "
        f"train=0..{TRAIN - 1} test={TRAIN}..{2 * TRAIN - 1} "
        f"K4/R1+two-FP16-scales={body_rate:.6f} bits/scalar"
    )

    labels = (
        "K4/V2 Q8",
        "exact-K/R1",
        "exact-K/R2",
        "row-K4/exact-V",
        "row-K4/R1",
        "row-K4/R2",
    )
    errors = {label: [] for label in labels}
    cache = {}
    region = slice(TRAIN, 2 * TRAIN)

    for layer, head, count, first, scale, _, query, _ in queries:
        identity = layer, head, count, first
        if identity not in cache:
            original_k = half(stack(keys, layer, head, count)[first:])
            original_v = half(stack(values, layer, head, count)[first:])
            transform = fit_map(original_k, original_v)

            control_k = original_k.copy()
            control_v = original_v.copy()
            control_k[region], control_v[region], _, _ = encode_pair(
                original_k[region], original_v[region], 8
            )

            quantized_k = midrise_rows(original_k[region], 4)
            exact_prediction = np.einsum(
                "ti,ij->tj", original_k[region], transform
            )
            quantized_prediction = np.einsum(
                "ti,ij->tj", quantized_k, transform
            )
            exact_residual = original_v[region] - exact_prediction
            quantized_residual = original_v[region] - quantized_prediction

            variants = {
                "K4/V2 Q8": (control_k, control_v),
                "exact-K/R1": (original_k, original_v.copy()),
                "exact-K/R2": (original_k, original_v.copy()),
            }
            variants["exact-K/R1"][1][region] = (
                exact_prediction + sign_rows(exact_residual)
            )
            variants["exact-K/R2"][1][region] = (
                exact_prediction + midrise_rows(exact_residual, 2)
            )
            for residual_label, residual in (
                ("exact-V", None),
                ("R1", sign_rows(quantized_residual)),
                ("R2", midrise_rows(quantized_residual, 2)),
            ):
                candidate_k = original_k.copy()
                candidate_v = original_v.copy()
                candidate_k[region] = quantized_k
                candidate_v[region] = (
                    original_v[region]
                    if residual_label == "exact-V"
                    else quantized_prediction + residual
                )
                variants[f"row-K4/{residual_label}"] = (
                    candidate_k, candidate_v
                )
            cache[identity] = original_k, original_v, variants

        original_k, original_v, variants = cache[identity]
        reference = attention(original_k, original_v, query, scale)
        for label, (candidate_k, candidate_v) in variants.items():
            errors[label].append(relative_error(
                attention(candidate_k, candidate_v, query, scale), reference
            ))

    for label in labels:
        summarize(label, errors[label])


if __name__ == "__main__":
    main()

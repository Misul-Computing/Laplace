#!/usr/bin/env python3
"""Screen fixed K4/V2 after an optimistic causal coordinate normalization."""

import argparse

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack
from prototype_kvarn_official import even_transform_rows, half, store_even_transform_rows
from prototype_syndrome_embedding import encode_pair


TILE = 128
TRAIN = 64
K_LEVELS = np.asarray([
    -2.73, -2.07, -1.62, -1.26, -0.94, -0.66, -0.39, -0.13,
    0.13, 0.39, 0.66, 0.94, 1.26, 1.62, 2.07, 2.73,
], dtype=np.float32)
V_LEVELS = np.asarray([-1.51, -0.45, 0.45, 1.51], dtype=np.float32)


def nearest(values, levels):
    values = np.asarray(values)
    flat = values.reshape(-1)
    output = levels[np.argmin(np.abs(flat[:, None] - levels[None]), axis=1)]
    return output.reshape(values.shape)


def coordinate_scales(matrix, pairwise):
    scale = np.sqrt(np.mean(matrix[:TRAIN].astype(np.float64) ** 2, axis=0))
    if pairwise:
        scale = np.repeat(
            np.sqrt(np.mean(scale.reshape(-1, 2) ** 2, axis=1)), 2
        )
    return np.maximum(scale, 1e-8).astype(np.float32)


def quantize_region(matrix, start, scales, levels, rotate):
    output = half(matrix).astype(np.float32)
    normalized = matrix[start:].astype(np.float32) / scales
    if rotate:
        normalized = store_even_transform_rows(normalized).astype(np.float32)
    decoded = nearest(normalized, levels)
    if rotate:
        decoded = even_transform_rows(decoded).astype(np.float32)
    output[start:] = half(decoded * scales).astype(np.float32)
    return output


def kvarn_reference(keys, values, start):
    output_k = half(keys).astype(np.float32)
    output_v = half(values).astype(np.float32)
    for offset in range(start, len(keys), TILE):
        output_k[offset:offset + TILE], output_v[offset:offset + TILE], _, _ = (
            encode_pair(keys[offset:offset + TILE], values[offset:offset + TILE], 8)
        )
    return output_k, output_v


def attention(keys, values, query, scale):
    weights = softmax(np.einsum("td,d->t", keys, query) * scale)
    return np.einsum("t,td->d", weights, values)


def summary(values):
    values = np.asarray(values) * 100.0
    return (
        f"mean={np.mean(values):8.3f}% "
        f"p95={np.percentile(values, 95):8.3f}% "
        f"max={np.max(values):8.3f}%"
    )


def evaluate(keys, values, queries, start):
    errors = {name: [] for name in (
        "reference/exact", "causal-coordinate/exact",
        "causal-coordinate/reference", "causal-coordinate-WH/exact",
        "causal-coordinate-WH/reference",
    )}
    cache = {}
    for layer, head, count, first, scale, _, query, _ in queries:
        identity = layer, head, count, first
        if identity not in cache:
            original_k = half(stack(keys, layer, head, count)[first:]).astype(np.float32)
            original_v = half(stack(values, layer, head, count)[first:]).astype(np.float32)
            k_scale = coordinate_scales(original_k, pairwise=True)
            v_scale = coordinate_scales(original_v, pairwise=False)
            reference = kvarn_reference(original_k, original_v, start)
            plain = (
                quantize_region(original_k, start, k_scale, K_LEVELS, False),
                quantize_region(original_v, start, v_scale, V_LEVELS, False),
            )
            rotated = (
                quantize_region(original_k, start, k_scale, K_LEVELS, True),
                quantize_region(original_v, start, v_scale, V_LEVELS, True),
            )
            cache[identity] = original_k, original_v, reference, plain, rotated

        original_k, original_v, reference, plain, rotated = cache[identity]
        exact = attention(original_k, original_v, query, scale)
        wanted = attention(*reference, query, scale)
        candidate = attention(*plain, query, scale)
        candidate_wh = attention(*rotated, query, scale)
        errors["reference/exact"].append(relative_error(wanted, exact))
        errors["causal-coordinate/exact"].append(relative_error(candidate, exact))
        errors["causal-coordinate/reference"].append(relative_error(candidate, wanted))
        errors["causal-coordinate-WH/exact"].append(relative_error(candidate_wh, exact))
        errors["causal-coordinate-WH/reference"].append(
            relative_error(candidate_wh, wanted)
        )

    print(
        f"traces={len(queries)} start={start} train={TRAIN} "
        "activation-derived coordinate scales metadata=free oracle"
    )
    for name, values_out in errors.items():
        print(f"{name:32s} {summary(values_out)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--start", type=int, default=128)
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    evaluate(keys, values, queries, args.start)


if __name__ == "__main__":
    main()

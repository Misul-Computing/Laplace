#!/usr/bin/env python3
"""Screen zero-metadata K4/V2 using the previous tile's exact scale state."""

import argparse

import numpy as np

from prototype_delta import (
    load_trace,
    relative_error,
    softmax,
    stack,
    variance_normalize,
)
from prototype_kvarn_official import (
    even_transform_rows,
    half,
    store_even_transform_rows,
)
from prototype_mixed_radix import quantize_tile


TILE = 128


def scale_state(source, levels, key):
    rotated = store_even_transform_rows(source)
    oriented = rotated.T if key else rotated
    balanced, columns, rows = variance_normalize(oriented, iterations=8)
    low = np.min(balanced, axis=1, keepdims=True)
    high = np.max(balanced, axis=1, keepdims=True)
    step = np.maximum((high - low) / (levels - 1), 1e-10)
    return columns, rows * step, rows * low


def decode_with_state(source, levels, key, state):
    rotated = store_even_transform_rows(source)
    oriented = rotated.T if key else rotated
    columns, scale, zero = state
    codes = np.clip(
        np.rint((oriented / columns - zero) / scale), 0, levels - 1
    )
    restored = (codes * scale + zero) * columns
    restored = restored.T if key else restored
    return half(even_transform_rows(restored)).astype(np.float64)


def attention(keys, values, query, scale):
    weights = softmax(np.einsum("td,d->t", keys, query) * scale)
    return np.einsum("t,td->d", weights, values)


def summarize(label, values):
    values = np.asarray(values) * 100.0
    print(
        f"{label:24s} mean={np.mean(values):8.3f}% "
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
    if context < 2 * TILE:
        raise ValueError("trace needs at least 256 tokens")

    labels = (
        "K4/V2 Q8 control",
        "previous-state K4/V2",
        "previous-state K4/exact-V",
        "exact-K/previous-state V2",
    )
    errors = {label: [] for label in labels}
    cache = {}
    region = slice(TILE, 2 * TILE)

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
            stale_k = decode_with_state(
                original_k[region], 16, True,
                scale_state(original_k[:TILE], 16, True),
            )
            stale_v = decode_with_state(
                original_v[region], 4, False,
                scale_state(original_v[:TILE], 4, False),
            )
            candidate_k = original_k.copy()
            candidate_v = original_v.copy()
            candidate_k[region] = stale_k
            candidate_v[region] = stale_v
            cache[identity] = (
                original_k,
                original_v,
                control_k,
                control_v,
                candidate_k,
                candidate_v,
            )

        (original_k, original_v, control_k, control_v,
         candidate_k, candidate_v) = cache[identity]
        reference = attention(original_k, original_v, query, scale)
        variants = {
            "K4/V2 Q8 control": (control_k, control_v),
            "previous-state K4/V2": (candidate_k, candidate_v),
            "previous-state K4/exact-V": (candidate_k, original_v),
            "exact-K/previous-state V2": (original_k, candidate_v),
        }
        for label, (candidate_keys, candidate_values) in variants.items():
            errors[label].append(relative_error(
                attention(candidate_keys, candidate_values, query, scale),
                reference,
            ))

    print(
        f"trace={args.trace} queries={len(queries)} "
        "first-tile exact state oracle second-tile K4/V2 metadata=zero"
    )
    for label in labels:
        summarize(label, errors[label])


if __name__ == "__main__":
    main()

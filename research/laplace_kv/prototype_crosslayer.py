#!/usr/bin/env python3
"""Test training-free cross-layer predictive KV coding."""

import argparse

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack, summarize


def ternary_rows(matrix):
    step = 1.224 * np.sqrt(np.mean(matrix * matrix, axis=1, keepdims=True))
    codes = np.zeros_like(matrix)
    codes[matrix > 0.5 * step] = 1.0
    codes[matrix < -0.5 * step] = -1.0
    return codes * step


def predict_tile(anchor, target, coordinate_scale):
    if coordinate_scale:
        denominator = np.sum(anchor * anchor, axis=0)
        multiplier = np.divide(
            np.sum(anchor * target, axis=0), denominator,
            out=np.zeros_like(denominator), where=denominator != 0,
        )
        multiplier = multiplier.astype(np.float16).astype(np.float64)
    else:
        denominator = np.sum(anchor * anchor)
        multiplier = np.sum(anchor * target) / denominator if denominator else 0.0
        multiplier = float(np.float16(multiplier))
    prediction = anchor * multiplier
    return prediction + ternary_rows(target - prediction)


def reconstruct(anchor, target, tile, mode, warmup):
    if mode == "linear":
        samples = min(warmup, target.shape[0])
        source = anchor[:samples].astype(np.float64)
        wanted = target[:samples].astype(np.float64)
        gram = np.einsum("nd,ne->de", source, source)
        ridge = 1e-3 * np.trace(gram) / max(1, gram.shape[0])
        transform = np.linalg.solve(
            gram + ridge * np.eye(gram.shape[0]),
            np.einsum("nd,ne->de", source, wanted),
        ).astype(np.float16).astype(np.float64)
        prediction = np.einsum(
            "nd,de->ne", anchor.astype(np.float64), transform
        )
        return prediction + ternary_rows(target.astype(np.float64) - prediction)
    output = np.empty_like(target, dtype=np.float64)
    sealed = target.shape[0] // tile * tile
    for start in range(0, sealed, tile):
        output[start:start + tile] = predict_tile(
            anchor[start:start + tile].astype(np.float64),
            target[start:start + tile].astype(np.float64),
            mode == "coordinate",
        )
    output[sealed:] = target[sealed:]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--groups", type=int, nargs="+", default=[4, 8, 12, 24])
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=64)
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    layers = sorted({record[0] for record in queries})
    errors = {(group, mode): [] for group in args.groups
              for mode in ("scalar", "coordinate", "linear")}
    cache = {}

    for layer, head, count, first, scale, _, query, _ in queries:
        target_k = stack(keys, layer, head, count)[first:]
        target_v = stack(values, layer, head, count)[first:]
        exact_weights = softmax(np.einsum("td,d->t", target_k, query) * scale)
        exact = np.einsum("t,td->d", exact_weights, target_v)
        layer_index = layers.index(layer)
        for group, mode in errors:
            anchor_layer = layers[layer_index // group * group]
            if anchor_layer == layer:
                continue
            key = anchor_layer, layer, head, count, first, group, mode
            if key not in cache:
                anchor_k = stack(keys, anchor_layer, head, count)[first:]
                anchor_v = stack(values, anchor_layer, head, count)[first:]
                cache[key] = (
                    reconstruct(anchor_k, target_k, args.tile, mode, args.warmup),
                    reconstruct(anchor_v, target_v, args.tile, mode, args.warmup),
                )
            candidate_k, candidate_v = cache[key]
            candidate_weights = softmax(
                np.einsum("td,d->t", candidate_k, query) * scale
            )
            candidate = np.einsum("t,td->d", candidate_weights, candidate_v)
            errors[group, mode].append(relative_error(candidate, exact))

    print(
        f"traces={len(queries)} layers={len(layers)} tile={args.tile} "
        "exact_anchor ternary_residual"
    )
    for group, mode in errors:
        summarize(f"g{group}/{mode}", errors[group, mode])


if __name__ == "__main__":
    main()

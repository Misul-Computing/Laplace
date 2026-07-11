#!/usr/bin/env python3
"""Test a fixed-size positive-random-feature attention state."""

import argparse
import math

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack, summarize


def gaussian_orthogonal_features(dim, count, seed):
    random = np.random.default_rng(seed)
    blocks = []
    while sum(block.shape[0] for block in blocks) < count:
        q, r = np.linalg.qr(random.standard_normal((dim, dim)))
        q *= np.sign(np.diag(r))[None, :]
        radii = np.sqrt(random.chisquare(dim, size=dim))
        blocks.append(q.T * radii[:, None])
    return np.concatenate(blocks, axis=0)[:count]


def log_features(vectors, omega, scale):
    vectors = vectors.astype(np.float64) * math.sqrt(scale)
    projection = np.einsum("nd,rd->nr", vectors, omega)
    penalty = 0.5 * np.sum(vectors * vectors, axis=1, keepdims=True)
    return projection - penalty


def state_attention(keys, values, query, scale, omega, tail):
    split = max(0, keys.shape[0] - tail)
    query_log = log_features(query[None, :], omega, scale)[0]
    log_weights = []
    denominators = []
    numerators = []
    if split:
        key_log = log_features(keys[:split], omega, scale)
        feature_max = np.max(key_log, axis=0)
        key_features = np.exp(key_log - feature_max)
        state_z = np.sum(key_features, axis=0)
        state_s = np.einsum(
            "nr,nd->rd", key_features, values[:split].astype(np.float64)
        )
        log_weights.extend(query_log + feature_max - math.log(omega.shape[0]))
        denominators.extend(state_z)
        numerators.extend(state_s)
    if split < keys.shape[0]:
        scores = np.einsum("td,d->t", keys[split:], query) * scale
        log_weights.extend(scores.astype(np.float64))
        denominators.extend(np.ones(scores.size))
        numerators.extend(values[split:].astype(np.float64))
    log_weights = np.asarray(log_weights)
    weights = np.exp(log_weights - np.max(log_weights))
    denominator = np.dot(weights, np.asarray(denominators))
    numerator = np.einsum("t,td->d", weights, np.asarray(numerators))
    return numerator / max(denominator, 1e-300)


def effective_bits(dim, context, features, tail):
    state_bytes = features * ((dim + 1) * 2 + 4)
    tail_bytes = min(context, tail) * 2 * dim * 2
    return (state_bytes + tail_bytes) * 8.0 / (context * 2 * dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--features", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--tails", type=int, nargs="+", default=[0, 64])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    dim = queries[0][6].size
    omegas = {
        count: gaussian_orthogonal_features(dim, count, args.seed)
        for count in args.features
    }
    errors = {(count, tail): [] for count in args.features for tail in args.tails}

    cache = {}
    for layer, head, count, first, scale, _, query, _ in queries:
        key = layer, head, count, first
        if key not in cache:
            cache[key] = (
                stack(keys, layer, head, count)[first:],
                stack(values, layer, head, count)[first:],
            )
        k, v = cache[key]
        exact_weights = softmax(np.einsum("td,d->t", k, query) * scale)
        exact = np.einsum("t,td->d", exact_weights, v)
        for (features, tail), result in errors.items():
            candidate = state_attention(
                k, v, query, scale, omegas[features], tail
            )
            result.append(relative_error(candidate, exact))

    context = queries[0][2] - queries[0][3]
    print(
        f"traces={len(queries)} context={context} dim={dim} "
        "fp64_oracle fp16_storage_estimate"
    )
    for features, tail in errors:
        bits = effective_bits(dim, context, features, tail)
        summarize(f"r{features}/t{tail}/{bits:.2f}b", errors[features, tail])


if __name__ == "__main__":
    main()

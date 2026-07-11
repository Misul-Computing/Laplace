#!/usr/bin/env python3
"""Measure an impossible query-aware upper bound for FP16 KV retention."""

import argparse

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.05, 0.10, 0.18, 0.25, 0.50]
    )
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    errors = {fraction: [] for fraction in args.fractions}
    masses = {fraction: [] for fraction in args.fractions}

    cache = {}
    for layer, head, count, first, scale, _, query, _ in queries:
        key = layer, head, count, first
        if key not in cache:
            cache[key] = (
                stack(keys, layer, head, count)[first:],
                stack(values, layer, head, count)[first:],
            )
        k, v = cache[key]
        scores = np.einsum("td,d->t", k, query) * scale
        weights = softmax(scores)
        exact = np.einsum("t,td->d", weights, v)
        order = np.argsort(scores)[::-1]
        for fraction in args.fractions:
            retained = max(1, int(np.floor(k.shape[0] * fraction)))
            selected = order[:retained]
            candidate_weights = softmax(scores[selected])
            candidate = np.einsum("t,td->d", candidate_weights, v[selected])
            errors[fraction].append(relative_error(candidate, exact))
            masses[fraction].append(np.sum(weights[selected]))

    print(f"traces={len(queries)} query-aware-oracle fp16-survivors")
    for fraction in args.fractions:
        bits = 16.0 * fraction
        summarize(f"keep{fraction:.2f}/{bits:.2f}b", errors[fraction])
        mass = np.asarray(masses[fraction]) * 100.0
        print(
            f"{'attention mass':18s} mean={np.mean(mass):7.3f}% "
            f"p05={np.percentile(mass, 5):7.3f}% min={np.min(mass):7.3f}%"
        )


if __name__ == "__main__":
    main()

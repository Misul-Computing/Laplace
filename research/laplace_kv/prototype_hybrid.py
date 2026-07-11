#!/usr/bin/env python3
"""Test an optimistic survivor plus full-moment attention bound."""

import argparse
import math

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack, summarize


def combine(selected_scores, selected_values, group_score, group_value):
    scores = np.append(selected_scores.astype(np.float64), group_score)
    values = np.vstack((selected_values.astype(np.float64), group_value))
    return np.einsum("t,td->d", softmax(scores), values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10, 0.18])
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    errors = {(fraction, mode): [] for fraction in args.fractions
              for mode in ("centroid", "full-moment")}
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
            removed = order[retained:]
            mean_k = np.mean(k[removed].astype(np.float64), axis=0)
            mean_v = np.mean(v[removed].astype(np.float64), axis=0)
            base_score = (
                np.dot(mean_k, query) * scale + math.log(removed.size)
            )
            centroid = combine(scores[selected], v[selected], base_score, mean_v)
            errors[fraction, "centroid"].append(relative_error(centroid, exact))

            delta_k = k[removed].astype(np.float64) - mean_k
            delta_v = v[removed].astype(np.float64) - mean_v
            centered_scores = np.einsum(
                "td,d->t", delta_k, query.astype(np.float64)
            ) * scale
            variance = np.mean(centered_scores * centered_scores)
            correction = 1.0 + 0.5 * variance
            cross = np.mean(delta_v * centered_scores[:, None], axis=0)
            moment = combine(
                scores[selected], v[selected],
                base_score + math.log(correction),
                mean_v + cross / correction,
            )
            errors[fraction, "full-moment"].append(relative_error(moment, exact))

    print(f"traces={len(queries)} query-aware-survivors full-moment-oracle")
    for fraction, mode in errors:
        summarize(f"keep{fraction:.2f}/{mode}", errors[fraction, mode])


if __name__ == "__main__":
    main()

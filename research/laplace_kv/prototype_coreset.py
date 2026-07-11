#!/usr/bin/env python3
"""Test query-independent moment coresets on captured KV traces."""

import argparse
import math

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack, summarize


def align(value, boundary=64):
    return (value + boundary - 1) // boundary * boundary


def kmeans(keys, clusters, iterations=8):
    mean = np.mean(keys, axis=0)
    chosen = [int(np.argmin(np.sum((keys - mean) ** 2, axis=1)))]
    distance = np.sum((keys - keys[chosen[0]]) ** 2, axis=1)
    while len(chosen) < clusters:
        chosen.append(int(np.argmax(distance)))
        distance = np.minimum(
            distance, np.sum((keys - keys[chosen[-1]]) ** 2, axis=1)
        )
    centers = keys[chosen].astype(np.float64)
    labels = np.zeros(keys.shape[0], dtype=np.int32)
    for _ in range(iterations):
        distances = np.sum((keys[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        nearest = distances[np.arange(keys.shape[0]), labels]
        for cluster in range(clusters):
            members = labels == cluster
            if np.any(members):
                centers[cluster] = np.mean(keys[members], axis=0)
            else:
                replacement = int(np.argmax(nearest))
                labels[replacement] = cluster
                centers[cluster] = keys[replacement]
                nearest[replacement] = 0.0
    return labels


def summarize_tile(keys, values, clusters, mode):
    labels = kmeans(keys.astype(np.float64), clusters)
    records = []
    for cluster in range(clusters):
        members = labels == cluster
        k = keys[members].astype(np.float64)
        v = values[members].astype(np.float64)
        mean_k = np.mean(k, axis=0)
        mean_v = np.mean(v, axis=0)
        delta_k = k - mean_k
        delta_v = v - mean_v
        variance = np.mean(delta_k * delta_k, axis=0)
        covariance = np.mean(delta_k * delta_v, axis=0)
        record = {
            "count": int(np.sum(members)),
            "k": mean_k.astype(np.float16).astype(np.float64),
            "v": mean_v.astype(np.float16).astype(np.float64),
        }
        if mode == "isotropic":
            record["variance"] = float(
                np.float16(np.mean(variance)).astype(np.float64)
            )
            record["covariance"] = covariance.astype(np.float16).astype(np.float64)
        elif mode == "diagonal":
            record["variance"] = variance.astype(np.float16).astype(np.float64)
            record["covariance"] = covariance.astype(np.float16).astype(np.float64)
        records.append(record)
    return records


def build_coreset(keys, values, tile, clusters, mode):
    records = []
    sealed = keys.shape[0] // tile * tile
    for start in range(0, sealed, tile):
        records.extend(
            summarize_tile(
                keys[start:start + tile], values[start:start + tile], clusters, mode
            )
        )
    for token in range(sealed, keys.shape[0]):
        records.append({"count": 1, "k": keys[token], "v": values[token]})
    return records


def attend(query, records, scale, mode):
    scores = []
    output_values = []
    query64 = query.astype(np.float64)
    query_norm = np.dot(query64, query64)
    for record in records:
        score = np.dot(record["k"], query64) * scale + math.log(record["count"])
        value = record["v"]
        if "covariance" in record:
            if mode == "isotropic":
                variance = scale * scale * query_norm * record["variance"]
            else:
                variance = scale * scale * np.dot(
                    query64 * query64, record["variance"]
                )
            correction = 1.0 + 0.5 * max(variance, 0.0)
            score += math.log(correction)
            value = value + scale * query64 * record["covariance"] / correction
        scores.append(score)
        output_values.append(value)
    return np.einsum("t,td->d", softmax(scores), np.stack(output_values))


def bits_per_scalar(dim, tile, clusters, mode):
    if mode == "centroid":
        bytes_per_record = 4 * dim + 2
    elif mode == "isotropic":
        bytes_per_record = 6 * dim + 4
    else:
        bytes_per_record = 8 * dim + 2
    return align(clusters * bytes_per_record) * 8.0 / (tile * 2 * dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--clusters", type=int, nargs="+", default=[4, 6, 8])
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    errors = {}
    caches = {}
    for clusters in args.clusters:
        for mode in ("centroid", "isotropic", "diagonal"):
            errors[clusters, mode] = []

    for layer, head, count, first, scale, _, query, _ in queries:
        key = (layer, head, count, first)
        if key not in caches:
            k = stack(keys, layer, head, count)[first:]
            v = stack(values, layer, head, count)[first:]
            variants = {}
            for clusters, mode in errors:
                variants[clusters, mode] = build_coreset(
                    k, v, args.tile, clusters, mode
                )
            caches[key] = k, v, variants
        k, v, variants = caches[key]
        exact_weights = softmax(np.einsum("td,d->t", k, query) * scale)
        exact = np.einsum("t,td->d", exact_weights, v)
        for candidate, result in errors.items():
            clusters, mode = candidate
            result.append(
                relative_error(attend(query, variants[candidate], scale, mode), exact)
            )

    dim = queries[0][6].size
    print(f"traces={len(queries)} tile={args.tile} dim={dim} fp16_summaries")
    for clusters, mode in errors:
        bits = bits_per_scalar(dim, args.tile, clusters, mode)
        summarize(f"m{clusters}/{mode}/{bits:.3f}b", errors[clusters, mode])


if __name__ == "__main__":
    main()

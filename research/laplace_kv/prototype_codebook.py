#!/usr/bin/env python3
"""Test a causal shared value codebook with symmetric low-bit keys."""

import argparse
from collections import defaultdict

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack


def quantize_symmetric(matrix, bits):
    peak = (1 << (bits - 1)) - 1
    scale = np.max(np.abs(matrix), axis=1, keepdims=True) / peak
    scale = np.maximum(scale, 1e-10)
    codes = np.clip(np.rint(matrix / scale), -peak, peak)
    return (codes * scale).astype(np.float32)


def train_codebook(samples, size, iterations=20):
    samples = samples.astype(np.float64)
    rng = np.random.default_rng(1)
    centers = np.empty((size, samples.shape[1]), dtype=np.float64)
    centers[0] = samples[0]
    distance = np.sum((samples - centers[0]) ** 2, axis=1)
    for index in range(1, size):
        total = np.sum(distance)
        choice = rng.integers(samples.shape[0]) if total == 0 else rng.choice(
            samples.shape[0], p=distance / total
        )
        centers[index] = samples[choice]
        distance = np.minimum(
            distance, np.sum((samples - centers[index]) ** 2, axis=1)
        )
    for _ in range(iterations):
        distances = np.sum(
            (samples[:, None, :] - centers[None, :, :]) ** 2, axis=2
        )
        labels = np.argmin(distances, axis=1)
        updated = centers.copy()
        for index in range(size):
            selected = samples[labels == index]
            if selected.size:
                updated[index] = np.mean(selected, axis=0)
        if np.array_equal(updated, centers):
            break
        centers = updated
    return centers.astype(np.float16).astype(np.float32)


def encode_values(values, codebook, width=4):
    groups = values.reshape(-1, width).astype(np.float64)
    distances = np.sum(
        (groups[:, None, :] - codebook[None, :, :]) ** 2, axis=2
    )
    decoded = codebook[np.argmin(distances, axis=1)]
    return decoded.reshape(values.shape).astype(np.float32)


def attention(query, scale, keys, values):
    scores = np.einsum("td,d->t", keys, query) * scale
    weights = softmax(scores)
    return np.einsum("t,td->d", weights, values)


def summarize(name, errors):
    errors = np.asarray(errors) * 100.0
    print(
        f"{name:15s} mean={np.mean(errors):8.3f}% "
        f"p95={np.percentile(errors, 95):8.3f}% "
        f"max={np.max(errors):8.3f}% pass={np.mean(errors <= 2.0) * 100:6.2f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--position", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=64)
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    final = defaultdict(list)
    for record in queries:
        if record[2] == args.position:
            final[record[0], record[1]].append(record)

    sizes = (16, 32, 64, 256)
    errors = {size: [] for size in sizes}
    value_only = {size: [] for size in sizes}
    key_only_errors = []

    for (layer, head), records in final.items():
        original_k = stack(keys, layer, head, args.position)
        original_v = stack(values, layer, head, args.position)
        keys4 = quantize_symmetric(original_k, 4)
        samples = original_v[:args.warmup].reshape(-1, 4)
        decoded = {}
        for size in sizes:
            codebook = train_codebook(samples, size)
            decoded[size] = encode_values(original_v, codebook)
        for record in records:
            query = record[6].astype(np.float64)
            scale = record[4]
            reference = attention(query, scale, original_k, original_v)
            key_only = attention(query, scale, keys4, original_v)
            key_only_errors.append(relative_error(key_only, reference))
            for size in sizes:
                candidate = attention(query, scale, keys4, decoded[size])
                exact_key = attention(query, scale, original_k, decoded[size])
                errors[size].append(relative_error(candidate, reference))
                value_only[size].append(relative_error(exact_key, reference))

    print(f"traces={len(key_only_errors)} warmup={args.warmup} group=4")
    summarize("K4/exact V", key_only_errors)
    for size in sizes:
        bits = int(np.log2(size)) / 4
        summarize(f"K4/VQ{size} {bits:g}b", errors[size])
        summarize(f"exactK/VQ{size}", value_only[size])


if __name__ == "__main__":
    main()

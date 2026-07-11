#!/usr/bin/env python3
"""Test causal query/output subspace correction on captured KV traces."""

import argparse
from collections import defaultdict

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack
from prototype_oracle import quantize_rows


def basis(samples, rank):
    _, _, vectors = np.linalg.svd(samples.astype(np.float64), full_matrices=False)
    return vectors[:rank].astype(np.float32)


def quantize_coefficients(coefficients, bits, group=64):
    output = np.empty_like(coefficients)
    for start in range(0, coefficients.shape[0], group):
        block = coefficients[start:start + group]
        output[start:start + block.shape[0]] = quantize_rows(block.T, bits).T
    return output


def corrected(base, original, vectors, bits):
    coefficients = np.einsum("td,rd->tr", original - base, vectors)
    if bits:
        coefficients = quantize_coefficients(coefficients, bits)
    return base + np.einsum("tr,rd->td", coefficients, vectors)


def attention(query, scale, keys, values):
    scores = np.einsum("td,d->t", keys, query) * scale
    weights = softmax(scores)
    return np.einsum("t,td->d", weights, values)


def summarize(name, errors):
    errors = np.asarray(errors) * 100.0
    print(
        f"{name:17s} mean={np.mean(errors):8.3f}% "
        f"p95={np.percentile(errors, 95):8.3f}% "
        f"max={np.max(errors):8.3f}% pass={np.mean(errors <= 2.0) * 100:6.2f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--position", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--coefficient-bits", type=int, default=0)
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)

    history = defaultdict(list)
    for record in queries:
        history[record[0], record[1]].append(record)

    ranks = (4, 8, 16, 32)
    results = {rank: [] for rank in ranks}
    key_only = {rank: [] for rank in ranks}
    value_only = {rank: [] for rank in ranks}
    base_errors = []
    cache = {}

    for (layer, head), records in history.items():
        past = [record for record in records if record[2] <= args.warmup]
        final = [record for record in records if record[2] == args.position]
        if not past or not final:
            continue
        query_samples = np.stack([record[6] for record in past])
        output_samples = np.stack([record[7] for record in past])
        original_k = stack(keys, layer, head, args.position)
        original_v = stack(values, layer, head, args.position)
        keys2 = quantize_rows(original_k, 2)
        values2 = quantize_rows(original_v, 2)

        variants = {}
        for rank in ranks:
            key_basis = basis(query_samples, rank)
            value_basis = basis(output_samples, rank)
            variants[rank] = (
                corrected(keys2, original_k, key_basis, args.coefficient_bits),
                corrected(values2, original_v, value_basis,
                          args.coefficient_bits),
            )
        cache[layer, head] = variants

        for record in final:
            query = record[6].astype(np.float64)
            scale = record[4]
            reference = attention(query, scale, original_k, original_v)
            base = attention(query, scale, keys2, values2)
            base_errors.append(relative_error(base, reference))
            for rank, (corrected_k, corrected_v) in variants.items():
                both = attention(query, scale, corrected_k, corrected_v)
                only_k = attention(query, scale, corrected_k, original_v)
                only_v = attention(query, scale, original_k, corrected_v)
                results[rank].append(relative_error(both, reference))
                key_only[rank].append(relative_error(only_k, reference))
                value_only[rank].append(relative_error(only_v, reference))

    print(
        f"traces={len(base_errors)} warmup={args.warmup} "
        f"coefficient_bits={args.coefficient_bits or 'fp32'}"
    )
    summarize("base K2/V2", base_errors)
    for rank in ranks:
        summarize(f"rank{rank} both", results[rank])
        summarize(f"rank{rank} key/exactV", key_only[rank])
        summarize(f"exactK/rank{rank} val", value_only[rank])


if __name__ == "__main__":
    main()

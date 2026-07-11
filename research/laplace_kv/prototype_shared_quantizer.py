#!/usr/bin/env python3
"""Screen fixed and prompt-shared metadata-free KV quantizers."""

import argparse

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack
from prototype_kvarn_official import (
    even_transform_rows,
    half,
    store_even_transform_rows,
)
from prototype_syndrome_embedding import encode_pair


TILE = 128
TRAIN = 64
TABLE_BYTES = 40  # 16 K and 4 V FP16 reconstruction levels.

GAUSSIAN16 = np.asarray([
    -2.73, -2.07, -1.62, -1.26, -0.94, -0.66, -0.39, -0.13,
    0.13, 0.39, 0.66, 0.94, 1.26, 1.62, 2.07, 2.73,
], dtype=np.float32)
GAUSSIAN4 = np.asarray([-1.51, -0.45, 0.45, 1.51], dtype=np.float32)


def align64(byte_count):
    return (byte_count + 63) // 64 * 64


def packed_rate(tokens, dim, reduced_k, table_bytes=0, anchor_bytes=0):
    body_bits = tokens * (6 * dim - reduced_k)
    packed = align64((body_bits + 7) // 8 + table_bytes + anchor_bytes)
    return packed, packed * 8 / (2 * tokens * dim)


def fixed_levels(bits, limit, mu):
    companded = np.linspace(-1.0, 1.0, 1 << bits)
    return (
        np.sign(companded)
        * np.expm1(np.abs(companded) * np.log1p(mu * limit))
        / mu
    ).astype(np.float32)


def nearest(values, levels):
    flat = np.asarray(values).reshape(-1)
    decoded = levels[np.argmin(np.abs(flat[:, None] - levels[None]), axis=1)]
    return decoded.reshape(np.asarray(values).shape)


def quantize_columns(matrix, full_levels, short_levels, short_columns):
    output = nearest(matrix, full_levels)
    if short_columns:
        output[:, :short_columns] = nearest(
            matrix[:, :short_columns], short_levels
        )
    return output


def lloyd_1d(samples, size, iterations=24):
    samples = np.asarray(samples, dtype=np.float64).ravel()
    centers = np.quantile(samples, (np.arange(size) + 0.5) / size)
    for _ in range(iterations):
        labels = np.argmin(np.abs(samples[:, None] - centers[None]), axis=1)
        updated = centers.copy()
        for index in range(size):
            selected = samples[labels == index]
            if selected.size:
                updated[index] = np.mean(selected)
        if np.array_equal(updated, centers):
            break
        centers = updated
    return half(np.sort(centers)).astype(np.float32)


def rotate(matrix):
    return store_even_transform_rows(matrix).astype(np.float32)


def restore(matrix):
    return half(even_transform_rows(matrix)).astype(np.float32)


def quantized_region(
    rotated, original, start, levels, short_levels, short_columns
):
    output = rotated.copy()
    output[start:] = quantize_columns(
        rotated[start:], levels, short_levels, short_columns
    )
    output = restore(output)
    output[:start] = half(original[:start]).astype(np.float32)
    return output


def fixed_candidates(keys, values, start):
    dim = keys.shape[1]
    key_anchor = keys[0].copy()
    value_anchor = half(values[0]).astype(np.float32)
    k16 = fixed_levels(4, 256.0, 15.0)
    k8 = fixed_levels(3, 256.0, 15.0)
    v4 = fixed_levels(2, 16.0, 7.0)

    raw_k = rotate(keys)
    raw_v = rotate(values)
    key_centered = rotate(keys - key_anchor)
    value_centered = rotate(values - value_anchor)
    return {
        "fixed": (
            quantized_region(raw_k, keys, start, k16, k8, 0),
            quantized_region(raw_v, values, start, v4, v4, 0),
            None,
        ),
        "fixed+K-gauge": (
            quantized_region(
                key_centered, keys - key_anchor, start, k16, k8, 0
            ),
            quantized_region(raw_v, values, start, v4, v4, 0),
            None,
        ),
        "fixed+KV-gauge": (
            quantized_region(
                key_centered, keys - key_anchor, start,
                k16, k8, dim // 8,
            ),
            quantized_region(
                value_centered, values - value_anchor, start, v4, v4, 0
            ),
            value_anchor,
        ),
    }


def shared_candidates(keys, values, start):
    dim = keys.shape[1]
    key_anchor = keys[0].copy()
    value_anchor = half(values[0]).astype(np.float32)
    variants = {}
    for name, source_k, source_v, correction, short_columns in (
        ("causal-shared", keys, values, None, 5),
        ("causal-shared+K-gauge", keys - key_anchor, values, None, 5),
        (
            "causal-shared+KV-gauge",
            keys - key_anchor,
            values - value_anchor,
            value_anchor,
            dim // 8 + 3,
        ),
    ):
        rotated_k = rotate(source_k)
        rotated_v = rotate(source_v)
        k16 = lloyd_1d(rotated_k[:TRAIN], 16)
        v4 = lloyd_1d(rotated_v[:TRAIN], 4)
        k8 = k16[np.rint(np.linspace(0, 15, 8)).astype(np.int64)]
        variants[name] = (
            quantized_region(
                rotated_k, source_k, start, k16, k8, short_columns
            ),
            quantized_region(rotated_v, source_v, start, v4, v4, 0),
            correction,
        )
    return variants


def shared_scale_candidate(keys, values, start):
    rotated_k = rotate(keys)
    rotated_v = rotate(values)
    key_scale = float(half(np.sqrt(np.mean(rotated_k[:TRAIN] ** 2))))
    value_scale = float(half(np.sqrt(np.mean(rotated_v[:TRAIN] ** 2))))
    k16 = GAUSSIAN16 * key_scale
    k8 = k16[np.rint(np.linspace(0, 15, 8)).astype(np.int64)]
    v4 = GAUSSIAN4 * value_scale
    return (
        quantized_region(rotated_k, keys, start, k16, k8, 1),
        quantized_region(rotated_v, values, start, v4, v4, 0),
        None,
    )


def reference(keys, values, start):
    output_k = keys.copy()
    output_v = values.copy()
    for tile in range(start, len(keys), TILE):
        output_k[tile:tile + TILE], output_v[tile:tile + TILE], _, _ = (
            encode_pair(
                keys[tile:tile + TILE], values[tile:tile + TILE], 8
            )
        )
    return output_k, output_v


def attention(query, scale, keys, values, correction=None):
    weights = softmax(np.einsum("td,d->t", keys, query) * scale)
    output = np.einsum("t,td->d", weights, values)
    return output if correction is None else output + correction


def summary(values):
    values = np.asarray(values) * 100.0
    return (
        f"mean={np.mean(values):8.3f}% "
        f"p95={np.percentile(values, 95):8.3f}% "
        f"max={np.max(values):8.3f}%"
    )


def evaluate(keys, values, queries, start):
    against_reference = {}
    against_exact = {}
    reference_errors = []
    gauge_errors = []
    cache = {}
    for layer, head, count, first, scale, _, query, _ in queries:
        identity = layer, head, count, first
        if identity not in cache:
            original_k = half(stack(keys, layer, head, count)[first:]).astype(
                np.float32
            )
            original_v = half(stack(values, layer, head, count)[first:]).astype(
                np.float32
            )
            variants = fixed_candidates(original_k, original_v, start)
            variants.update(shared_candidates(original_k, original_v, start))
            variants["causal-RMS"] = shared_scale_candidate(
                original_k, original_v, start
            )
            cache[identity] = (
                original_k,
                original_v,
                reference(original_k, original_v, start),
                variants,
            )
        original_k, original_v, (reference_k, reference_v), variants = (
            cache[identity]
        )
        query = query.astype(np.float64)
        exact = attention(query, scale, original_k, original_v)
        baseline = attention(query, scale, reference_k, reference_v)
        reference_errors.append(relative_error(baseline, exact))

        key_anchor = original_k[0]
        value_anchor = half(original_v[0]).astype(np.float32)
        gauge = attention(
            query,
            scale,
            original_k - key_anchor,
            original_v - value_anchor,
            value_anchor,
        )
        gauge_errors.append(relative_error(gauge, exact))
        for name, (candidate_k, candidate_v, correction) in variants.items():
            candidate = attention(
                query, scale, candidate_k, candidate_v, correction
            )
            against_reference.setdefault(name, []).append(
                relative_error(candidate, baseline)
            )
            against_exact.setdefault(name, []).append(
                relative_error(candidate, exact)
            )
    return reference_errors, gauge_errors, against_reference, against_exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    if not queries or queries[0][6].size % 16:
        raise ValueError("trace must use a head dimension divisible by 16")
    if any(record[2] != queries[0][2] for record in queries):
        raise ValueError("trace must contain one evaluation position")
    tokens = queries[0][2] - queries[0][3]
    dim = queries[0][6].size
    if tokens % TILE:
        raise ValueError("trace token count must be divisible by 128")

    print(
        f"traces={len(queries)} heads={len(set((x[0], x[1]) for x in queries))} "
        f"tokens={tokens} dim={dim} train={TRAIN}"
    )
    layouts = (
        ("fixed", 0, 0, 0),
        ("fixed+K-gauge", 0, 0, 0),
        ("fixed+KV-gauge", dim // 8, 0, 2 * dim),
        ("causal-RMS", 1, 4, 0),
        ("causal-shared", 5, TABLE_BYTES, 0),
        ("causal-shared+K-gauge", 5, TABLE_BYTES, 0),
        (
            "causal-shared+KV-gauge",
            dim // 8 + 3,
            TABLE_BYTES,
            2 * dim,
        ),
    )

    for label, start in (("full-prefix", 0), ("sealed-tile", TILE)):
        encoded_tokens = tokens - start
        print(f"\n[{label}] quantized_tokens={encoded_tokens}")
        for name, reduced, table, anchor in layouts:
            packed, rate = packed_rate(
                encoded_tokens, dim, reduced, table, anchor
            )
            if rate > 3.0:
                raise RuntimeError(f"{name} exceeds three bits per scalar")
            print(
                f"storage {name:27s} bytes={packed:6d} "
                f"bits/scalar={rate:.6f}"
            )
        reference_errors, gauge_errors, versus_reference, versus_exact = (
            evaluate(keys, values, queries, start)
        )
        print(f"K4/V2-Q8 vs exact          {summary(reference_errors)}")
        print(f"exact gauge identity       {summary(gauge_errors)}")
        for name in versus_reference:
            print(f"{name:27s} vs-ref {summary(versus_reference[name])}")
            print(f"{'':27s} vs-exact {summary(versus_exact[name])}")
    if max(gauge_errors) > 1e-5:
        raise RuntimeError("attention gauge identity failed")


if __name__ == "__main__":
    main()

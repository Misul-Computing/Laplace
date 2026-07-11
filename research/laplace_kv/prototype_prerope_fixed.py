#!/usr/bin/env python3
"""Screen fixed-rate rotated product quantization before and after RoPE."""

import argparse

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack
from prototype_even_transform import factored_transform
from prototype_fixed_product import fixed_gaussian_codebook, reconstruct_table


def rope(values, positions, base, inverse=False):
    output = np.asarray(values, dtype=np.float64).copy()
    dimension = output.shape[-1]
    half = dimension // 2
    frequencies = np.power(base, -2.0 * np.arange(half) / dimension)
    angles = np.asarray(positions, dtype=np.float64)[..., None] * frequencies
    if inverse:
        angles = -angles
    cosine = np.cos(angles)
    sine = np.sin(angles)
    left = output[..., :half].copy()
    right = output[..., half:2 * half].copy()
    output[..., :half] = left * cosine - right * sine
    output[..., half:2 * half] = left * sine + right * cosine
    return output


def log8_scale(matrix):
    scale = np.sqrt(np.mean(matrix * matrix, axis=1, keepdims=True))
    code = np.clip(
        np.rint((np.log2(np.maximum(scale, 2.0 ** -8)) + 8) * 16),
        0,
        255,
    )
    return np.exp2(code / 16 - 8)


def nearest(batch, table):
    output = np.empty_like(batch)
    for start in range(0, len(batch), 4096):
        current = batch[start:start + 4096]
        distances = np.sum(
            (current[:, None] - table[None]) ** 2,
            axis=2,
        )
        output[start:start + len(current)] = table[np.argmin(distances, axis=1)]
    return output


def quantize_values(matrix, table):
    scale = log8_scale(matrix)
    normalized = factored_transform(matrix) / scale
    decoded = nearest(normalized.reshape(-1, 2), table).reshape(matrix.shape)
    return factored_transform(decoded * scale).astype(np.float64)


def quantize_keys(matrix, table7, table8):
    """Use exactly the K rate left after V2 and two log8 vector scales."""
    dimension = matrix.shape[1]
    pairs = dimension // 2
    high_count = max(0, pairs - 16)
    scale = log8_scale(matrix)
    normalized = (factored_transform(matrix) / scale).reshape(-1, pairs, 2)
    decoded = np.empty_like(normalized)
    if high_count:
        high = np.floor(np.arange(pairs) * high_count / pairs) != np.floor(
            (np.arange(pairs) + 1) * high_count / pairs
        )
    else:
        high = np.zeros(pairs, dtype=bool)
    decoded[:, ~high] = nearest(
        normalized[:, ~high].reshape(-1, 2), table7
    ).reshape(normalized.shape[0], np.count_nonzero(~high), 2)
    if high_count:
        decoded[:, high] = nearest(
            normalized[:, high].reshape(-1, 2), table8
        ).reshape(normalized.shape[0], high_count, 2)
    restored = factored_transform(decoded.reshape(matrix.shape) * scale)
    key_code_bits = pairs * 7 + high_count
    effective = (key_code_bits + pairs * 4 + 16) / dimension / 2
    return restored.astype(np.float64), effective


def attention(keys, values, query, scale):
    weights = softmax(np.einsum("td,d->t", keys, query) * scale)
    return np.einsum("t,td->d", weights, values)


def summarize(label, errors):
    values = np.asarray(errors) * 100.0
    print(
        f"{label:24s} mean={np.mean(values):8.3f}% "
        f"p95={np.percentile(values, 95):8.3f}% max={np.max(values):8.3f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--rope-base", type=float, default=1e6)
    parser.add_argument("--anchor", type=int, default=128)
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    dimension = queries[0][6].size
    if dimension % 32:
        raise ValueError("diagnostic requires a head dimension divisible by 32")

    code7, scale7 = fixed_gaussian_codebook(128, 417)
    code8, scale8 = fixed_gaussian_codebook(256, 419)
    value_code, value_scale = fixed_gaussian_codebook(16, 418)
    table7 = reconstruct_table(code7, scale7).astype(np.float64)
    table8 = reconstruct_table(code8, scale8).astype(np.float64)
    value_table = reconstruct_table(value_code, value_scale).astype(np.float64)

    labels = (
        "post-RoPE",
        "post-RoPE exact V",
        "post-RoPE gauge",
        "pre-RoPE",
        "pre-RoPE exact V",
        "pre-RoPE centered",
        "exact K / V2",
    )
    errors = {label: [] for label in labels}
    cache = {}
    rate = None
    for layer, head, count, first, scale, _, query, _ in queries:
        identity = layer, head, count, first
        if identity not in cache:
            original_k = stack(keys, layer, head, count)[first:].astype(np.float64)
            original_v = stack(values, layer, head, count)[first:].astype(np.float64)
            positions = np.arange(first, count)
            post_k, rate = quantize_keys(original_k, table7, table8)
            post_v = quantize_values(original_v, value_table)

            anchor_count = min(args.anchor, len(original_k))
            key_anchor = np.mean(original_k[:anchor_count], axis=0)
            value_anchor = np.mean(original_v[:anchor_count], axis=0)
            gauge_k, _ = quantize_keys(original_k - key_anchor, table7, table8)
            gauge_v = quantize_values(original_v - value_anchor, value_table)

            raw_k = rope(original_k, positions, args.rope_base, inverse=True)
            pre_k, _ = quantize_keys(raw_k, table7, table8)
            pre_k = rope(pre_k, positions, args.rope_base)
            raw_anchor = np.mean(raw_k[:anchor_count], axis=0)
            pre_centered, _ = quantize_keys(
                raw_k - raw_anchor, table7, table8
            )
            pre_centered = rope(
                pre_centered + raw_anchor, positions, args.rope_base
            )
            cache[identity] = (
                original_k,
                original_v,
                post_k,
                post_v,
                gauge_k,
                gauge_v,
                value_anchor,
                pre_k,
                pre_centered,
            )

        (original_k, original_v, post_k, post_v, gauge_k, gauge_v,
         value_anchor, pre_k, pre_centered) = cache[identity]
        query = query.astype(np.float64)
        reference = attention(original_k, original_v, query, scale)
        candidates = {
            "post-RoPE": attention(post_k, post_v, query, scale),
            "post-RoPE exact V": attention(post_k, original_v, query, scale),
            "post-RoPE gauge": attention(gauge_k, gauge_v, query, scale)
            + value_anchor,
            "pre-RoPE": attention(pre_k, post_v, query, scale),
            "pre-RoPE exact V": attention(pre_k, original_v, query, scale),
            "pre-RoPE centered": attention(pre_centered, post_v, query, scale),
            "exact K / V2": attention(original_k, post_v, query, scale),
        }
        for label, candidate in candidates.items():
            errors[label].append(relative_error(candidate, reference))

    print(
        f"dim={dimension} queries={len(queries)} effective_bits={rate:.6f} "
        "table_bytes_not_counted=816"
    )
    for label in labels:
        summarize(label, errors[label])


if __name__ == "__main__":
    main()

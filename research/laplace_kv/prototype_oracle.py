#!/usr/bin/env python3
"""Optimistic mixed-precision bound for a sub-3-bit LaplaceKV candidate."""

import argparse

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack


def quantize_rows(matrix, bits):
    levels = (1 << bits) - 1
    low = np.min(matrix, axis=1, keepdims=True)
    high = np.max(matrix, axis=1, keepdims=True)
    step = np.maximum((high - low) / levels, 1e-10)
    codes = np.clip(np.rint((matrix - low) / step), 0, levels)
    return (codes * step + low).astype(np.float32)


def quantize_ternary_rows(matrix):
    magnitude = np.abs(matrix)
    level = np.mean(magnitude, axis=1, keepdims=True)
    for _ in range(8):
        selected = magnitude > 0.5 * level
        level = (
            np.sum(magnitude * selected, axis=1, keepdims=True)
            / np.maximum(np.sum(selected, axis=1, keepdims=True), 1)
        )
    return (np.sign(matrix) * (magnitude > 0.5 * level) * level).astype(np.float32)


def attention(scores, values):
    weights = softmax(scores)
    return weights, np.einsum("t,td->d", weights, values)


def oracle_refine(query, scale, keys2, values2, keys8, values8,
                  reference, upgrades):
    query = query.astype(np.float64)
    scores = np.einsum("td,d->t", keys2, query) * scale
    values = values2.astype(np.float64).copy()
    scores8 = np.einsum("td,d->t", keys8, query) * scale
    key_done = np.zeros(keys2.shape[0], dtype=bool)
    value_done = np.zeros(keys2.shape[0], dtype=bool)

    for _ in range(upgrades):
        weights, output = attention(scores, values)
        current = relative_error(output, reference)
        reference_norm = max(np.linalg.norm(reference), 1e-20)
        ratios = np.exp(np.clip(scores8 - scores, -20, 20))
        changes = weights * (ratios - 1.0)
        key_candidates = (
            output[None, :] + changes[:, None] * values
        ) / (1.0 + changes)[:, None]
        key_errors = np.linalg.norm(
            key_candidates - reference[None, :], axis=1
        ) / reference_norm
        key_errors[key_done] = np.inf

        value_candidates = output[None, :] + weights[:, None] * (
            values8 - values
        )
        value_errors = np.linalg.norm(
            value_candidates - reference[None, :], axis=1
        ) / reference_norm
        value_errors[value_done] = np.inf

        key_token = int(np.argmin(key_errors))
        value_token = int(np.argmin(value_errors))
        if min(key_errors[key_token], value_errors[value_token]) >= current:
            break
        if key_errors[key_token] <= value_errors[value_token]:
            scores[key_token] = scores8[key_token]
            key_done[key_token] = True
        else:
            values[value_token] = values8[value_token]
            value_done[value_token] = True

    _, output = attention(scores, values)
    return relative_error(output, reference), np.sum(key_done), np.sum(value_done)


def summarize(name, errors):
    errors = np.asarray(errors) * 100.0
    print(
        f"{name:12s} mean={np.mean(errors):8.3f}% "
        f"p95={np.percentile(errors, 95):8.3f}% "
        f"max={np.max(errors):8.3f}% pass={np.mean(errors <= 2.0) * 100:6.2f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--target-bits", type=float, default=3.0)
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    cache = {}
    base_errors = []
    three_errors = []
    asymmetric_errors = []
    ternary_errors = []
    value2_errors = []
    oracle_errors = []
    key_upgrades = 0
    value_upgrades = 0

    for layer, head, count, first, scale, _, query, _ in queries:
        key = layer, head, count, first
        if key not in cache:
            original_k = stack(keys, layer, head, count)[first:]
            original_v = stack(values, layer, head, count)[first:]
            cache[key] = (
                original_k,
                original_v,
                quantize_rows(original_k, 2),
                quantize_rows(original_v, 2),
                quantize_rows(original_k, 3),
                quantize_rows(original_v, 3),
                quantize_rows(original_k, 4),
                quantize_rows(original_k, 8),
                quantize_rows(original_v, 8),
                quantize_ternary_rows(original_v),
            )
        (original_k, original_v, keys2, values2, keys3, values3, keys4,
         keys8, values8, values_ternary) = cache[key]
        query64 = query.astype(np.float64)
        exact_scores = np.einsum("td,d->t", original_k, query64) * scale
        _, reference = attention(exact_scores, original_v)
        base_scores = np.einsum("td,d->t", keys2, query64) * scale
        _, base = attention(base_scores, values2)
        base_errors.append(relative_error(base, reference))
        scores3 = np.einsum("td,d->t", keys3, query64) * scale
        _, output3 = attention(scores3, values3)
        three_errors.append(relative_error(output3, reference))
        scores42 = np.einsum("td,d->t", keys4, query64) * scale
        _, output42 = attention(scores42, values2)
        asymmetric_errors.append(relative_error(output42, reference))
        _, output4t = attention(scores42, values_ternary)
        ternary_errors.append(relative_error(output4t, reference))
        scores82 = np.einsum("td,d->t", keys8, query64) * scale
        _, output82 = attention(scores82, values2)
        value2_errors.append(relative_error(output82, reference))

        # This spends the complete bit budget on 2->8 upgrades and gives
        # scales, indexes, and alignment away for free. It is an upper bound,
        # not a realizable format.
        upgrade_budget = int(((args.target_bits - 2.0) * 2 * original_k.size) //
                             (6 * original_k.shape[1]))
        error, upgraded_k, upgraded_v = oracle_refine(
            query64, scale, keys2, values2, keys8, values8,
            reference, upgrade_budget
        )
        oracle_errors.append(error)
        key_upgrades += upgraded_k
        value_upgrades += upgraded_v

    print(f"traces={len(queries)} target_bits={args.target_bits:g} metadata=free")
    summarize("base K2/V2", base_errors)
    summarize("base K3/V3", three_errors)
    summarize("base K4/V2", asymmetric_errors)
    summarize("base K4/Vtern", ternary_errors)
    summarize("base K8/V2", value2_errors)
    summarize("oracle", oracle_errors)
    print(f"oracle upgrades: K={key_upgrades} V={value_upgrades}")


if __name__ == "__main__":
    main()

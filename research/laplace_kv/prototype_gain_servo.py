#!/usr/bin/env python3
"""Screen a zero-metadata symbol-clocked K4/V2 gain servo."""

import argparse
from statistics import NormalDist

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack
from prototype_kvarn_official import (
    even_transform_rows,
    half,
    store_even_transform_rows,
)
from prototype_mixed_radix import quantize_tile


TILE = 128


def gaussian_recipe(bits):
    clipping = {2: 1.5, 4: 2.7}[bits]
    peak = (1 << bits) / 2.0 - 0.5
    step = clipping / peak
    normal = NormalDist()
    samples = np.asarray([
        normal.inv_cdf((index + 0.5) / 65536)
        for index in range(65536)
    ])
    levels = np.clip(
        np.rint(samples / step + peak), 0, (1 << bits) - 1
    ) - peak
    return step, float(np.sqrt(np.mean(levels * levels)))


def servo_rows(source, reference, bits, servo, feedback):
    transformed = store_even_transform_rows(source).astype(np.float64)
    reference = store_even_transform_rows(reference).astype(np.float64)
    step, target = gaussian_recipe(bits)
    base = np.sqrt(np.mean(reference * reference)) * step
    base = max(float(half(base)), 2.0 ** -24)
    peak = (1 << bits) / 2.0 - 0.5
    gain = 0
    error = np.zeros(source.shape[1], dtype=np.float64)
    decoded = np.empty_like(transformed)
    low_threshold = target * 2.0 ** (-1.0 / 16.0)
    high_threshold = target * 2.0 ** (1.0 / 16.0)
    for token, row in enumerate(transformed):
        scale = base * 2.0 ** (gain / 8.0)
        wanted = row + feedback * error
        codes = np.clip(
            np.rint(wanted / scale + peak), 0, (1 << bits) - 1
        )
        levels = codes - peak
        decoded[token] = levels * scale
        error = wanted - decoded[token]
        if servo:
            occupancy = np.sqrt(np.mean(levels * levels))
            if occupancy > high_threshold:
                gain = min(gain + 1, 63)
            elif occupancy < low_threshold:
                gain = max(gain - 1, -64)
    return half(even_transform_rows(decoded)).astype(np.float64)


def attention(keys, values, query, scale):
    weights = softmax(np.einsum("td,d->t", keys, query) * scale)
    return np.einsum("t,td->d", weights, values)


def summarize(label, values):
    values = np.asarray(values) * 100.0
    print(
        f"{label:25s} mean={np.mean(values):8.3f}% "
        f"p95={np.percentile(values, 95):8.3f}% "
        f"max={np.max(values):8.3f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    context = max(item[2] - item[3] for item in queries)
    queries = [item for item in queries if item[2] - item[3] == context]
    if context < 2 * TILE:
        raise ValueError("trace needs at least 256 tokens")

    labels = (
        "K4/V2 Q8 control",
        "previous fixed",
        "previous servo",
        "previous servo+feedback",
        "current-RMS fixed oracle",
        "current-RMS servo oracle",
    )
    errors = {label: [] for label in labels}
    cache = {}
    region = slice(TILE, 2 * TILE)

    for layer, head, count, first, scale, _, query, _ in queries:
        identity = layer, head, count, first
        if identity not in cache:
            original_k = half(stack(keys, layer, head, count)[first:])
            original_v = half(stack(values, layer, head, count)[first:])
            control_k = original_k.copy()
            control_v = original_v.copy()
            control_k[region], _ = quantize_tile(
                original_k[region], 16, True, 8
            )
            control_v[region], _ = quantize_tile(
                original_v[region], 4, False, 8
            )
            variants = {"K4/V2 Q8 control": (control_k, control_v)}
            for label, reference, servo, feedback in (
                ("previous fixed", slice(0, TILE), False, 0.0),
                ("previous servo", slice(0, TILE), True, 0.0),
                ("previous servo+feedback", slice(0, TILE), True, 0.5),
                ("current-RMS fixed oracle", region, False, 0.0),
                ("current-RMS servo oracle", region, True, 0.0),
            ):
                candidate_k = original_k.copy()
                candidate_v = original_v.copy()
                candidate_k[region] = servo_rows(
                    original_k[region], original_k[reference], 4,
                    servo, feedback,
                )
                candidate_v[region] = servo_rows(
                    original_v[region], original_v[reference], 2,
                    servo, feedback,
                )
                variants[label] = candidate_k, candidate_v
            cache[identity] = original_k, original_v, variants

        original_k, original_v, variants = cache[identity]
        reference = attention(original_k, original_v, query, scale)
        for label, (candidate_k, candidate_v) in variants.items():
            errors[label].append(relative_error(
                attention(candidate_k, candidate_v, query, scale), reference
            ))

    print(
        f"trace={args.trace} queries={len(queries)} "
        "K4/V2 body=3.000000 bits/scalar state=decoder-synchronized"
    )
    for label in labels:
        summarize(label, errors[label])


if __name__ == "__main__":
    main()

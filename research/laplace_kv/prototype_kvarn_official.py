#!/usr/bin/env python3
"""Independent reference for the released dense KVarN tile format."""

import argparse

import numpy as np

from prototype_delta import (
    hadamard_rows,
    load_trace,
    relative_error,
    softmax,
    stack,
    summarize,
    variance_normalize,
)


def half(values):
    return np.asarray(values).astype(np.float16).astype(np.float64)


def scale_field(values, bits=8):
    values = np.asarray(values, dtype=np.float64)
    low = float(np.min(values))
    high = float(np.max(values))
    maximum = (1 << bits) - 1
    step = max((high - low) / maximum, 1e-10)
    low = float(half(low))
    step = max(float(half(step)), 2.0 ** -24)
    codes = np.clip(np.rint((values - low) / step), 0, maximum)
    return codes * step + low


def binary_float(values, min_exponent, max_exponent, mantissa_bits,
                 signed=True):
    values = np.asarray(values, dtype=np.float64)
    signs = np.sign(values) if signed else np.ones_like(values)
    magnitude = np.abs(values) if signed else np.maximum(values, 0.0)
    subnormal_step = 2.0 ** (min_exponent - mantissa_bits)
    exponent = np.floor(np.log2(np.maximum(magnitude, subnormal_step)))
    exponent = np.clip(exponent, min_exponent, max_exponent)
    step = np.where(
        magnitude < 2.0 ** min_exponent,
        subnormal_step,
        2.0 ** (exponent - mantissa_bits),
    )
    maximum = (2.0 - 2.0 ** -mantissa_bits) * 2.0 ** max_exponent
    rounded = np.minimum(np.rint(magnitude / step) * step, maximum)
    return signs * rounded


def polarity_fp8(values, signed):
    if signed:
        return binary_float(values, -6, 7, 3, True)
    return binary_float(values, -14, 15, 3, False)


def store_hadamard_rows(matrix):
    width = matrix.shape[1]
    output = half(matrix).copy()
    stride = 1
    while stride < width:
        blocks = output.reshape(-1, stride * 2)
        left = blocks[:, :stride].copy()
        right = blocks[:, stride:].copy()
        blocks[:, :stride] = left + right
        blocks[:, stride:] = left - right
        stride *= 2
    return half(output * half(1.0 / np.sqrt(width)))


def store_even_transform_rows(matrix):
    source = np.asarray(matrix)
    width = source.shape[1]
    power = width & -width
    if power == width:
        return store_hadamard_rows(source)
    blocks = source.reshape(source.shape[0], width // power, power)
    output = np.stack(
        [store_hadamard_rows(blocks[:, block])
         for block in range(blocks.shape[1])],
        axis=1,
    )
    output -= (2.0 / blocks.shape[1]) * np.sum(
        output, axis=1, keepdims=True
    )
    return half(output).reshape(source.shape)


def even_transform_rows(matrix):
    source = np.asarray(matrix, dtype=np.float64)
    width = source.shape[1]
    power = width & -width
    blocks = source.reshape(source.shape[0], width // power, power)
    output = np.stack(
        [hadamard_rows(blocks[:, block])
         for block in range(blocks.shape[1])],
        axis=1,
    )
    if blocks.shape[1] > 1:
        output -= (2.0 / blocks.shape[1]) * np.sum(
            output, axis=1, keepdims=True
        )
    return output.reshape(source.shape)


def quantize_tile(source, bits, key, compact_metadata=0):
    rotated = store_even_transform_rows(source)
    oriented = rotated.T if key else rotated
    balanced, columns, rows = variance_normalize(oriented, iterations=8)
    low = np.min(balanced, axis=1, keepdims=True)
    high = np.max(balanced, axis=1, keepdims=True)
    step = np.maximum((high - low) / ((1 << bits) - 1), 1e-10)
    codes = np.clip(np.rint((balanced - low) / step), 0, (1 << bits) - 1)
    absorbed_scale = rows * step
    absorbed_zero = rows * low
    other_scale = columns
    if compact_metadata == "fp8":
        metadata_scale = lambda values: polarity_fp8(values, False)
        metadata_zero = lambda values: polarity_fp8(values, True)
    elif compact_metadata:
        metadata_scale = metadata_zero = (
            lambda values: scale_field(values, compact_metadata)
        )
    else:
        metadata_scale = metadata_zero = half
    restored = (
        codes * metadata_scale(absorbed_scale) + metadata_zero(absorbed_zero)
    ) * metadata_scale(other_scale)
    restored = restored.T if key else restored
    return half(even_transform_rows(restored)).astype(np.float32)


def quantize(matrix, bits, key, group, sink):
    output = matrix.copy()
    for start in range(sink, matrix.shape[0] - group + 1, group):
        output[start:start + group] = quantize_tile(
            matrix[start:start + group], bits, key
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--sink", type=int, default=128)
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    modes = ((4, 2), (2, 2))
    errors = {mode: [] for mode in modes}
    cache = {}
    for layer, head, count, first, scale, _, query, _ in queries:
        key = layer, head, count, first
        if key not in cache:
            original_k = stack(keys, layer, head, count)[first:]
            original_v = stack(values, layer, head, count)[first:]
            variants = {
                mode: (
                    quantize(original_k, mode[0], True, args.group, args.sink),
                    quantize(original_v, mode[1], False, args.group, args.sink),
                )
                for mode in modes
            }
            cache[key] = original_k, original_v, variants
        original_k, original_v, variants = cache[key]
        exact_weights = softmax(np.einsum("td,d->t", original_k, query) * scale)
        exact = np.einsum("t,td->d", exact_weights, original_v)
        for mode, result in errors.items():
            candidate_k, candidate_v = variants[mode]
            weights = softmax(np.einsum("td,d->t", candidate_k, query) * scale)
            candidate = np.einsum("t,td->d", weights, candidate_v)
            result.append(relative_error(candidate, exact))

    print(
        f"traces={len(queries)} group={args.group} sink={args.sink} "
        "iterations=8 metadata=fp16"
    )
    for key_bits, value_bits in modes:
        summarize(f"K{key_bits}/V{value_bits}", errors[key_bits, value_bits])


if __name__ == "__main__":
    main()

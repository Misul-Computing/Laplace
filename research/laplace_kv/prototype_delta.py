#!/usr/bin/env python3
"""Evaluate direct differential attention on a LaplaceKV trace."""

import argparse
import math
import struct

import numpy as np


HEADER = struct.Struct("<IIiiiiifi")
MAGIC = 0x4C4B5654


def load_trace(path):
    keys = {}
    values = {}
    queries = []
    with open(path, "rb") as trace:
        while header := trace.read(HEADER.size):
            if len(header) != HEADER.size:
                raise ValueError("truncated trace header")
            magic, kind, layer, head, position, dim, first, scale, index = (
                HEADER.unpack(header)
            )
            if magic != MAGIC or kind not in (1, 2, 3):
                raise ValueError("invalid trace record")
            count = dim * (2 if kind == 3 else 1)
            payload = np.fromfile(trace, dtype="<f4", count=count)
            if payload.size != count:
                raise ValueError("truncated trace payload")
            if kind == 1:
                keys[layer, head, position] = payload
            elif kind == 2:
                values[layer, head, position] = payload
            else:
                queries.append(
                    (layer, head, position, first, scale, index,
                     payload[:dim], payload[dim:])
                )
    return keys, values, queries


def stack(records, layer, head, count):
    return np.stack([records[layer, head, token] for token in range(count)])


def dpcm(matrix, levels):
    reconstructed = np.empty_like(matrix)
    deltas = np.empty_like(matrix)
    steps = np.zeros(matrix.shape[0], dtype=np.float32)
    reconstructed[0] = matrix[0].astype(np.float16).astype(np.float32)
    deltas[0] = reconstructed[0]
    peak = max(abs(levels[0]), abs(levels[-1]))
    for token in range(1, matrix.shape[0]):
        residual = matrix[token] - reconstructed[token - 1]
        step = np.max(np.abs(residual)) / peak
        if step == 0:
            deltas[token] = 0
        else:
            scaled = residual / step
            indices = np.argmin(
                np.abs(scaled[:, None] - levels[None, :]), axis=1
            )
            deltas[token] = levels[indices] * step
        steps[token] = step
        reconstructed[token] = reconstructed[token - 1] + deltas[token]
    return reconstructed, deltas, steps


def ternary_dpcm(matrix):
    reconstructed = np.empty_like(matrix)
    deltas = np.empty_like(matrix)
    steps = np.zeros(matrix.shape[0], dtype=np.float32)
    reconstructed[0] = matrix[0].astype(np.float16).astype(np.float32)
    deltas[0] = reconstructed[0]
    for token in range(1, matrix.shape[0]):
        residual = matrix[token] - reconstructed[token - 1]
        step = 1.224 * np.sqrt(np.mean(residual * residual))
        codes = np.zeros(matrix.shape[1], dtype=np.float32)
        if step != 0:
            codes[residual > 0.5 * step] = 1
            codes[residual < -0.5 * step] = -1
        deltas[token] = codes * step
        steps[token] = step
        reconstructed[token] = reconstructed[token - 1] + deltas[token]
    return reconstructed, deltas, steps


def softmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.exp(scores - np.max(scores))
    return weights / np.sum(weights)


def direct_delta_attention(query, k_delta, v_delta, logit_scale, key_steps):
    scores = np.cumsum(
        np.einsum(
            "td,d->t", k_delta.astype(np.float64), query.astype(np.float64)
        )
    ) * logit_scale
    weights = softmax(scores)
    suffix = np.cumsum(weights[::-1])[::-1]
    output = np.einsum("t,td->d", suffix, v_delta)
    variance = np.sum(query * query) * key_steps * key_steps / 12.0
    corrected = softmax(scores - 0.5 * variance * logit_scale * logit_scale)
    corrected_output = np.einsum(
        "t,td->d", np.cumsum(corrected[::-1])[::-1], v_delta
    )
    return output, corrected_output


def imbalance(tile):
    columns = np.std(tile, axis=0, ddof=1)
    rows = np.std(tile, axis=1, ddof=1)
    return (
        np.max(columns) / max(np.min(columns), 1e-8)
        + np.max(rows) / max(np.min(rows), 1e-8)
    )


def variance_normalize(tile, iterations=16):
    matrix = tile.astype(np.float64)
    log_columns = np.zeros((1, matrix.shape[1]))
    log_rows = np.zeros((matrix.shape[0], 1))
    current = matrix.copy()
    best_imbalance = imbalance(current)
    best_columns = np.ones_like(log_columns)
    best_rows = np.ones_like(log_rows)
    for _ in range(iterations):
        column_std = np.clip(
            np.std(current, axis=0, ddof=1, keepdims=True), 1e-3, 1e3
        )
        log_columns = np.clip(
            log_columns + np.log(column_std), -0.3, 10.0
        )
        current = matrix / np.exp(log_columns) / np.exp(log_rows)
        row_std = np.clip(
            np.std(current, axis=1, ddof=1, keepdims=True), 1e-3, 1e3
        )
        log_rows = np.clip(log_rows + np.log(row_std), -0.3, 10.0)
        current = matrix / np.exp(log_columns) / np.exp(log_rows)
        score = imbalance(current)
        if score <= best_imbalance:
            best_imbalance = score
            best_columns = np.exp(log_columns).copy()
            best_rows = np.exp(log_rows).copy()
    return matrix / best_columns / best_rows, best_columns, best_rows


def asymmetric_rtn(tile, bits):
    low = np.min(tile, axis=1, keepdims=True)
    high = np.max(tile, axis=1, keepdims=True)
    step = np.maximum((high - low) / ((1 << bits) - 1), 1e-10)
    codes = np.clip(np.rint((tile - low) / step), 0, (1 << bits) - 1)
    return codes * step + low


def hadamard_rows(matrix):
    width = matrix.shape[1]
    if width == 0 or width & (width - 1):
        return matrix.astype(np.float64)
    output = matrix.astype(np.float64).copy()
    stride = 1
    while stride < width:
        blocks = output.reshape(-1, stride * 2)
        left = blocks[:, :stride].copy()
        right = blocks[:, stride:].copy()
        blocks[:, :stride] = left + right
        blocks[:, stride:] = left - right
        stride *= 2
    return output / math.sqrt(width)


def kvarn(matrix, bits, key, group=128):
    output = np.empty_like(matrix)
    for start in range(0, matrix.shape[0], group):
        source = matrix[start:start + group]
        rotated = hadamard_rows(source)
        oriented = rotated.T if key else rotated
        balanced, columns, rows = variance_normalize(oriented)
        restored = asymmetric_rtn(balanced, bits) * columns * rows
        restored = restored.T if key else restored
        output[start:start + source.shape[0]] = hadamard_rows(restored)
    return output


def relative_error(candidate, reference):
    return np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), 1e-20)


def summarize(name, values):
    values = np.asarray(values) * 100.0
    print(
        f"{name:18s} mean={np.mean(values):7.3f}% "
        f"p95={np.percentile(values, 95):7.3f}% max={np.max(values):7.3f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)

    modes = {
        "dpcm2": np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32),
        "dpcm3": np.arange(-3.5, 4.0, dtype=np.float32),
    }
    errors = {"current": []}
    for mode in modes:
        errors[mode] = []
        errors[mode + "+bias"] = []
    errors["dpcm2/ternary"] = []
    errors["dpcm2/ternary+bias"] = []
    for name in ("kvarn2/2", "kvarn3/2", "kvarn4/2", "kvarn4/4", "kvarn8/8"):
        errors[name] = []
    identity_errors = []

    cache = {}
    for layer, head, count, first, scale, _, query, current in queries:
        key = (layer, head, count, first)
        if key not in cache:
            k = stack(keys, layer, head, count)[first:]
            v = stack(values, layer, head, count)[first:]
            variants = {}
            for name, levels in modes.items():
                rk, dk, sk = dpcm(k, levels)
                rv, dv, _ = dpcm(v, levels)
                variants[name] = (rk, dk, sk, rv, dv)
            rk, dk, sk = dpcm(k, modes["dpcm2"])
            rv, dv, _ = ternary_dpcm(v)
            variants["dpcm2/ternary"] = (rk, dk, sk, rv, dv)
            kvarn_variants = {
                "kvarn2/2": (kvarn(k, 2, True), kvarn(v, 2, False)),
                "kvarn3/2": (kvarn(k, 3, True), kvarn(v, 2, False)),
                "kvarn4/2": (kvarn(k, 4, True), kvarn(v, 2, False)),
                "kvarn4/4": (kvarn(k, 4, True), kvarn(v, 4, False)),
                "kvarn8/8": (kvarn(k, 8, True), kvarn(v, 8, False)),
            }
            cache[key] = (k, v, variants, kvarn_variants)
        k, v, variants, kvarn_variants = cache[key]
        query64 = query.astype(np.float64)
        exact_scores = np.einsum(
            "td,d->t", k.astype(np.float64), query64
        ) * scale
        exact_weights = softmax(exact_scores)
        exact = np.einsum("t,td->d", exact_weights, v.astype(np.float64))
        errors["current"].append(relative_error(current, exact))
        for name, (rk, dk, sk, rv, dv) in variants.items():
            candidate, corrected = direct_delta_attention(
                query, dk, dv, scale, sk
            )
            identity_weights = softmax(
                np.einsum("td,d->t", rk, query) * scale
            )
            identity = np.einsum("t,td->d", identity_weights, rv)
            identity_errors.append(relative_error(candidate, identity))
            errors[name].append(relative_error(candidate, exact))
            errors[name + "+bias"].append(relative_error(corrected, exact))
        for name, (rk, rv) in kvarn_variants.items():
            weights = softmax(
                np.einsum("td,d->t", rk.astype(np.float64), query64) * scale
            )
            output = np.einsum("t,td->d", weights, rv)
            errors[name].append(relative_error(output, exact))

    print(f"traces={len(queries)}")
    for name, result in errors.items():
        summarize(name, result)
    identity_max = max(identity_errors)
    print(f"delta identity max={identity_max:.3e}")
    if identity_max > 1e-4:
        raise RuntimeError("direct delta attention identity failed")
    dim = queries[0][6].size
    print(
        "estimated fixed-rate bits/scalar at long context: "
        f"dpcm2={2 + 16 / dim:.3f}, "
        f"dpcm2/ternary={1.8 + 16 / dim:.3f} "
        "(five trits per byte)"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Screen frozen-metric mixed Q4/Q2 quantization in a Walsh basis."""

import argparse
import struct

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack
from prototype_eoptshrinkq import gaussian_lloyd, half, matmul, summarize
from prototype_syndrome_embedding import encode_pair
from prototype_weighted_spectral import hadamard


TILE = 128
DIMENSION = 64
SELECTED = 47
METRIC_HEADER = struct.Struct("<8s8Id")
TABLE_BYTES = (4 + 16) * 4
MASK_BYTES = 16
ALIGNMENT = 64
ALLOCATIONS = ((47, 0), (46, 0), (32, 15), (24, 23))


def align(value, multiple=ALIGNMENT):
    return (value + multiple - 1) // multiple * multiple


def load_metrics(path):
    with open(path, "rb") as source:
        payload = source.read()
    if len(payload) < METRIC_HEADER.size:
        raise ValueError("truncated weight-metric header")
    fields = METRIC_HEADER.unpack_from(payload)
    magic, version, layers, heads, dimension = fields[:5]
    if magic.rstrip(b"\0") != b"LWPSD1" or version != 2:
        raise ValueError("unsupported weight-metric file")
    if dimension != DIMENSION:
        raise ValueError("this screen requires D64 metrics")
    matrix_count = 2 * layers * heads * dimension * dimension
    spectrum_count = 2 * layers * heads * dimension
    expected = METRIC_HEADER.size + 8 * (
        matrix_count + spectrum_count + matrix_count
    )
    if len(payload) != expected:
        raise ValueError("weight-metric file size does not match its header")
    metrics = np.frombuffer(
        payload, dtype="<f8", count=matrix_count, offset=METRIC_HEADER.size
    ).reshape(2, layers, heads, dimension, dimension)
    if not np.all(np.isfinite(metrics)):
        raise ValueError("weight metrics contain non-finite values")
    return metrics


def metric_diagonals(metrics, rotation):
    # diag(H M H^T) without materializing every transformed matrix.
    return np.einsum("ia,klhab,ib->klhi", rotation, metrics, rotation)


def mask_for(diagonal, count):
    mask = np.zeros(DIMENSION, dtype=bool)
    if count:
        order = np.lexsort((np.arange(DIMENSION), -diagonal))
        mask[order[:count]] = True
    return mask


def selectors(diagonals):
    output = {}
    for key_count, value_count in ALLOCATIONS:
        name = f"metric-K{key_count}/V{value_count}"
        output[name] = np.empty(
            (diagonals.shape[1], diagonals.shape[2], 2, DIMENSION), dtype=bool
        )
        stripe = np.zeros_like(output[name])
        for layer in range(diagonals.shape[1]):
            for head in range(diagonals.shape[2]):
                output[name][layer, head, 0] = mask_for(
                    diagonals[0, layer, head], key_count
                )
                output[name][layer, head, 1] = mask_for(
                    diagonals[1, layer, head], value_count
                )
                stripe[layer, head, 0, :key_count] = True
                stripe[layer, head, 1, :value_count] = True
        output[f"stripe-K{key_count}/V{value_count}"] = stripe

    allocations = {}
    for total in (47, 46):
        global_masks = np.empty(
            (diagonals.shape[1], diagonals.shape[2], 2, DIMENSION), dtype=bool
        )
        global_stripes = np.zeros_like(global_masks)
        split = []
        for layer in range(diagonals.shape[1]):
            for head in range(diagonals.shape[2]):
                normalized = diagonals[:, layer, head].copy()
                normalized /= np.sum(normalized, axis=1, keepdims=True)
                combined = normalized.reshape(-1)
                order = np.lexsort((np.arange(2 * DIMENSION), -combined))
                selected = order[:total]
                masks = np.zeros(2 * DIMENSION, dtype=bool)
                masks[selected] = True
                masks = masks.reshape(2, DIMENSION)
                global_masks[layer, head] = masks
                key_count = int(np.count_nonzero(masks[0]))
                value_count = total - key_count
                global_stripes[layer, head, 0, :key_count] = True
                global_stripes[layer, head, 1, :value_count] = True
                split.append((key_count, value_count))
        output[f"metric-global{total}"] = global_masks
        output[f"stripe-global{total}-allocation"] = global_stripes
        allocations[total] = split
    return output, allocations


def quantize_rows(source, q2, q4, mask):
    source = np.asarray(source, dtype=np.float64)
    exact_norm = np.linalg.norm(source, axis=1)
    stored_norm = half(exact_norm).astype(np.float64)
    unit = np.zeros_like(source)
    nonzero = stored_norm != 0
    unit[nonzero] = source[nonzero] / stored_norm[nonzero, None]
    # Captured K/V rows are already in the engine's Walsh domain.
    transformed = unit
    decoded = q2[np.argmin(
        np.abs(transformed[:, :, None] - q2[None, None, :]), axis=2
    )]
    if np.any(mask):
        selected = transformed[:, mask]
        decoded[:, mask] = q4[np.argmin(
            np.abs(selected[:, :, None] - q4[None, None, :]), axis=2
        )]
    return decoded * stored_norm[:, None]


def build_variants(keys, values, selector_map, q2, q4):
    names = ("K4/V2-Q8", "TQ2", *selector_map)
    variants = {}
    for layer in range(selector_map[next(iter(selector_map))].shape[0]):
        for head in range(selector_map[next(iter(selector_map))].shape[1]):
            original_k = half(stack(keys, layer, head, 256))
            original_v = half(stack(values, layer, head, 256))
            per_group = {
                name: (original_k.copy(), original_v.copy()) for name in names
            }
            zero = np.zeros(DIMENSION, dtype=bool)
            for start in range(0, 256, TILE):
                region = slice(start, start + TILE)
                qk, qv, _, _ = encode_pair(
                    original_k[region], original_v[region], 8
                )
                per_group["K4/V2-Q8"][0][region] = qk
                per_group["K4/V2-Q8"][1][region] = qv
                per_group["TQ2"][0][region] = quantize_rows(
                    original_k[region], q2, q4, zero
                )
                per_group["TQ2"][1][region] = quantize_rows(
                    original_v[region], q2, q4, zero
                )
                for name, masks in selector_map.items():
                    per_group[name][0][region] = quantize_rows(
                        original_k[region], q2, q4,
                        masks[layer, head, 0],
                    )
                    per_group[name][1][region] = quantize_rows(
                        original_v[region], q2, q4,
                        masks[layer, head, 1],
                    )
            variants[layer, head] = original_k, original_v, per_group
    return variants


def evaluate(queries, variants, lifecycle):
    names = next(iter(variants.values()))[2].keys()
    errors = {name: [] for name in names}
    score_errors = {name: [] for name in names}
    reference_errors = {name: [] for name in names if name != "K4/V2-Q8"}
    for layer, head, count, _, scale, _, query, _ in queries:
        if lifecycle == "final" and count != 256:
            continue
        original_k, original_v, per_group = variants[layer, head]
        exact_k = original_k[:count]
        exact_v = original_v[:count]
        exact_scores = matmul(
            exact_k.astype(np.float64), query.astype(np.float64)
        ) * scale
        exact_weights = softmax(exact_scores)
        exact = matmul(exact_weights, exact_v.astype(np.float64))
        outputs = {}
        for name, (decoded_k, decoded_v) in per_group.items():
            sealed = (
                count if lifecycle == "final" or name != "K4/V2-Q8"
                else count // TILE * TILE
            )
            candidate_scores = exact_scores.copy()
            candidate_scores[:sealed] = matmul(
                decoded_k[:sealed].astype(np.float64), query.astype(np.float64)
            ) * scale
            weights = softmax(candidate_scores)
            candidate = matmul(
                weights[:sealed], decoded_v[:sealed].astype(np.float64)
            )
            if sealed < count:
                candidate += matmul(
                    weights[sealed:], exact_v[sealed:].astype(np.float64)
                )
            outputs[name] = candidate
            errors[name].append(relative_error(candidate, exact))
            score_errors[name].append(relative_error(candidate_scores, exact_scores))
        reference = outputs["K4/V2-Q8"]
        for name in reference_errors:
            reference_errors[name].append(relative_error(outputs[name], reference))
    return errors, score_errors, reference_errors


def print_results(title, result):
    errors, score_errors, reference_errors = result
    print(f"\n[{title}]")
    for name in errors:
        print(f"{name:27s} attention {summarize(errors[name])}")
        print(f"{'':27s} K-score  {summarize(score_errors[name])}")
        if name in reference_errors:
            print(f"{'':27s} vs K4/V2 {summarize(reference_errors[name])}")


def print_storage(layers, heads):
    print("\n[storage]")
    records = layers * heads
    for selected in (47, 46):
        row_bits = 2 * DIMENSION * 2 + selected * 2 + 2 * 16
        tile_body = TILE * row_bits // 8
        tile_raw = tile_body + MASK_BYTES
        tile_addressed = align(tile_raw)
        print(
            f"selected={selected} self-contained pair body+norms={tile_body} B "
            f"masks={MASK_BYTES} B raw={tile_raw} B aligned={tile_addressed} B "
            f"rate={tile_addressed * 8 / (2 * TILE * DIMENSION):.6f} bits/scalar"
        )
        print(
            f"  immediate packed cache: {row_bits} bits/token/head pair, "
            f"two masks={MASK_BYTES} B per layer/head, shared FP32 levels={TABLE_BYTES} B, "
            f"one {ALIGNMENT}-byte-aligned global pool; no mutable tail"
        )
        passing = None
        for context in range(1, 4097):
            raw = (records * context * row_bits + 7) // 8
            cache = align(raw + records * MASK_BYTES + TABLE_BYTES)
            rate = cache * 8 / (records * context * 2 * DIMENSION)
            if rate <= 3.0:
                passing = context
                break
        print(f"  first fully counted context <=3: {passing}")
        for context in (1, 64, 127, 128, 4095, 4096, 16383, 16384, 65535, 65536):
            raw = (records * context * row_bits + 7) // 8
            cache = align(raw + records * MASK_BYTES + TABLE_BYTES)
            rate = cache * 8 / (records * context * 2 * DIMENSION)
            print(
                f"  context={context:5d} raw-body={raw:9d} B "
                f"pool={cache:9d} B effective={rate:.6f} bits/scalar"
            )


def self_test(rotation, q2, q4):
    identity = matmul(rotation, rotation)
    if not np.allclose(identity, np.eye(DIMENSION), atol=1e-12):
        raise RuntimeError("Walsh transform is not orthonormal")
    source = np.arange(2 * DIMENSION, dtype=np.float64).reshape(2, DIMENSION)
    zero = np.zeros(DIMENSION, dtype=bool)
    if not np.array_equal(
        quantize_rows(source, q2, q4, zero),
        quantize_rows(source, q2, q4, zero),
    ):
        raise RuntimeError("quantizer is not deterministic")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="official all-token D64 trace")
    parser.add_argument("metrics", help="matching frozen-weight .lwm file")
    args = parser.parse_args()

    keys, values, queries = load_trace(args.trace)
    metrics = load_metrics(args.metrics)
    if len(queries) != 86016 or {item[2] for item in queries} != set(range(1, 257)):
        raise ValueError("requires the complete 256-token Qwen2.5 trace")
    if metrics.shape[:3] != (2, 24, 2):
        raise ValueError("trace and metric layer/head shapes differ")

    rotation = hadamard(DIMENSION)
    q2 = gaussian_lloyd(2, DIMENSION).astype(np.float32).astype(np.float64)
    q4 = gaussian_lloyd(4, DIMENSION).astype(np.float32).astype(np.float64)
    self_test(rotation, q2, q4)
    diagonals = metric_diagonals(metrics, rotation)
    selector_map, allocations = selectors(diagonals)
    print(
        f"trace={args.trace} queries={len(queries)} metrics={args.metrics} "
        f"candidate=stored-norm+WH+Gaussian-Q4/Q2 selected={SELECTED}"
    )
    for total, split in allocations.items():
        counts = np.asarray(split)
        print(
            f"global{total} allocation K mean={np.mean(counts[:, 0]):.3f} "
            f"min={np.min(counts[:, 0])} max={np.max(counts[:, 0])}; "
            f"V mean={np.mean(counts[:, 1]):.3f} "
            f"min={np.min(counts[:, 1])} max={np.max(counts[:, 1])}"
        )
    variants = build_variants(
        keys, values, selector_map, q2, q4
    )
    print_results(
        "causal all-token immediate candidate; K4/V2 completed tiles",
        evaluate(queries, variants, "causal"),
    )
    print_results(
        "current 256-token prefix, both tiles encoded",
        evaluate(queries, variants, "final"),
    )
    print_storage(metrics.shape[1], metrics.shape[2])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Screen frozen-weight KV metrics without prompt or query calibration."""

import argparse
from collections import defaultdict
import math

import gguf
import numpy as np

from prototype_delta import load_trace, relative_error, stack
from prototype_eoptshrinkq import (
    adaptive_lloyd,
    eopt_shrink,
    gaussian_lloyd,
    half,
    matmul,
    nearest_indices,
    pair_layout,
    quantized_factors,
    summarize,
    tq_mse,
)
from prototype_syndrome_embedding import encode_pair


TILE = 128
SUPPORT_RCOND = 1e-8


def hadamard(dimension):
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < dimension:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    if matrix.shape != (dimension, dimension):
        raise ValueError("head dimension must be a power of two")
    return matrix / math.sqrt(dimension)


def tensor_map(reader):
    return {tensor.name: tensor for tensor in reader.tensors}


def dequant(tensors, name):
    tensor = tensors.get(name)
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {name}")
    return np.asarray(
        gguf.dequantize(tensor.data, tensor.tensor_type), dtype=np.float64
    )


def metadata(reader, name):
    field = reader.fields.get(name)
    if field is None:
        raise KeyError(f"GGUF metadata not found: {name}")
    return field.contents()


def rope_average(matrix, dimension, base, horizon):
    """Average R_p M R_p^T for absolute positions 0 through P-1."""
    half_dimension = dimension // 2
    positions = np.arange(horizon, dtype=np.float64)[:, None]
    frequencies = base ** (
        -2.0 * np.arange(half_dimension, dtype=np.float64) / dimension
    )
    angles = positions * frequencies[None, :]
    cosine = np.cos(angles)
    sine = np.sin(angles)
    cc = matmul(cosine.T, cosine) / horizon
    sc = matmul(sine.T, cosine) / horizon
    cs = matmul(cosine.T, sine) / horizon
    ss = matmul(sine.T, sine) / horizon
    output = np.empty_like(matrix)
    for row in range(half_dimension):
        rows = (row, row + half_dimension)
        for column in range(half_dimension):
            columns = (column, column + half_dimension)
            a = matrix[rows[0], columns[0]]
            b = matrix[rows[0], columns[1]]
            c = matrix[rows[1], columns[0]]
            d = matrix[rows[1], columns[1]]
            output[rows[0], columns[0]] = (
                a * cc[row, column] - c * sc[row, column]
                - b * cs[row, column] + d * ss[row, column]
            )
            output[rows[0], columns[1]] = (
                a * cs[row, column] - c * ss[row, column]
                + b * cc[row, column] - d * sc[row, column]
            )
            output[rows[1], columns[0]] = (
                a * sc[row, column] + c * cc[row, column]
                - b * ss[row, column] - d * cs[row, column]
            )
            output[rows[1], columns[1]] = (
                a * ss[row, column] + c * cs[row, column]
                + b * sc[row, column] + d * cc[row, column]
            )
    return (output + output.T) * 0.5


def psd_metric(matrix):
    symmetric = (matrix + matrix.T) * 0.5
    values, vectors = np.linalg.eigh(symmetric)
    largest = max(float(values[-1]), 0.0)
    cutoff = largest * SUPPORT_RCOND
    keep = values > cutoff
    if not np.any(keep):
        raise ValueError("metric has no positive numerical support")
    clipped = np.where(keep, values, 0.0)
    scale = matrix.shape[0] / np.sum(clipped)
    clipped *= scale
    order = np.argsort(clipped)[::-1]
    clipped = clipped[order]
    vectors = vectors[:, order]
    support = clipped > 0
    condition = clipped[0] / clipped[support][-1]
    normalized = matmul(vectors * clipped, vectors.T)
    return {
        "matrix": normalized,
        "values": clipped,
        "vectors": vectors,
        "support": support,
        "condition": condition,
        "dropped": int(np.count_nonzero(~support)),
        "cutoff": cutoff,
    }


def build_metrics(reader, tensors, layers, dimension, horizon):
    q_heads = int(metadata(reader, "qwen2.attention.head_count"))
    kv_heads = int(metadata(reader, "qwen2.attention.head_count_kv"))
    hidden = int(metadata(reader, "qwen2.embedding_length"))
    rope_base = float(metadata(reader, "qwen2.rope.freq_base"))
    if q_heads * dimension != hidden or q_heads % kv_heads:
        raise ValueError("unexpected Qwen GQA geometry")
    gqa = q_heads // kv_heads
    transform = hadamard(dimension)
    metrics = {}
    weight_types = set()

    for layer in layers:
        prefix = f"blk.{layer}."
        q_tensor = tensors[prefix + "attn_q.weight"]
        o_tensor = tensors[prefix + "attn_output.weight"]
        weight_types.add((q_tensor.tensor_type.name, o_tensor.tensor_type.name))
        query_weight = dequant(tensors, prefix + "attn_q.weight")
        output_weight = dequant(tensors, prefix + "attn_output.weight")
        gain = dequant(tensors, prefix + "attn_norm.weight").reshape(-1)
        query_bias = dequant(tensors, prefix + "attn_q.bias").reshape(-1)
        if prefix + "attn_q_norm.weight" in tensors:
            raise ValueError(
                "nonlinear query RMSNorm needs a separate covariance derivation"
            )
        if query_weight.shape != (q_heads * dimension, hidden):
            raise ValueError("unexpected query-projection shape")
        if output_weight.shape != (hidden, q_heads * dimension):
            raise ValueError("unexpected output-projection shape")

        for kv_head in range(kv_heads):
            key_raw = np.zeros((dimension, dimension), dtype=np.float64)
            value_raw = np.zeros_like(key_raw)
            first_head = kv_head * gqa
            for local in range(gqa):
                query_head = first_head + local
                region = slice(
                    query_head * dimension, (query_head + 1) * dimension
                )
                block = query_weight[region] * gain[None, :]
                bias = query_bias[region]
                key_raw += matmul(block, block.T) + np.outer(bias, bias)
                output_block = output_weight[:, region]
                value_raw += matmul(output_block.T, output_block)

            key_position_averaged = rope_average(
                key_raw, dimension, rope_base, horizon
            )
            key_wh = matmul(matmul(transform, key_position_averaged), transform)
            value_wh = matmul(matmul(transform, value_raw), transform)
            metrics[layer, kv_head] = (
                psd_metric(key_wh), psd_metric(value_wh)
            )
    return metrics, {
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "gqa": gqa,
        "hidden": hidden,
        "rope_base": rope_base,
        "weight_types": sorted(weight_types),
    }


def weighted_rank(matrix, metric, rank):
    if rank == 0:
        return np.zeros_like(matrix)
    values = metric["values"]
    vectors = metric["vectors"]
    support = metric["support"]
    basis = vectors[:, support]
    root = matmul(basis * np.sqrt(values[support]), basis.T)
    inverse = matmul(basis / np.sqrt(values[support]), basis.T)
    weighted = matmul(matrix, root)
    left, singular, right = np.linalg.svd(weighted, full_matrices=False)
    return matmul(
        matmul(left[:, :rank] * singular[:rank], right[:rank]), inverse
    )


def truncated(matrix, rank):
    if rank == 0:
        return np.zeros_like(matrix)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    return matmul(left[:, :rank] * singular[:rank], right[:rank])


def encode_signal(matrix, signal, rank, rotation, centers):
    residual, _, _ = tq_mse(matrix - signal, rotation, centers)
    if rank == 0:
        return residual
    left, singular, right = np.linalg.svd(signal, full_matrices=False)
    correction = quantized_factors(
        left[:, :rank], singular[:rank], right[:rank]
    )
    return residual + correction


def quantize_fixed_basis(matrix, metric, rank, rotation, centers):
    base, _, _ = tq_mse(matrix, rotation, centers)
    if rank == 0:
        return base
    basis = half(metric["vectors"][:, :rank]).astype(np.float64)
    coefficients = matmul(matrix - base, basis)
    codebook = adaptive_lloyd(coefficients, 4)
    decoded = codebook[nearest_indices(coefficients, codebook)]
    return base + matmul(decoded, basis.T)


def mixed_precision(matrix, selected, base_centers, selected_centers):
    source = np.asarray(matrix, dtype=np.float64)
    stored_norms = half(np.linalg.norm(source, axis=1)).astype(np.float64)
    unit = np.zeros_like(source)
    nonzero = stored_norms != 0
    unit[nonzero] = source[nonzero] / stored_norms[nonzero, None]
    decoded = base_centers[nearest_indices(unit, base_centers)]
    if len(selected):
        selected = np.asarray(selected, dtype=np.int64)
        decoded[:, selected] = selected_centers[
            nearest_indices(unit[:, selected], selected_centers)
        ]
    return decoded * stored_norms[:, None]


def selected_coordinates(key_metric, value_metric, key_count, value_count):
    key_order = np.argsort(np.diag(key_metric["matrix"]))[::-1]
    value_order = np.argsort(np.diag(value_metric["matrix"]))[::-1]
    return key_order[:key_count], value_order[:value_count]


def jointly_selected(key_metric, value_metric, total):
    sensitivity = np.concatenate((
        np.diag(key_metric["matrix"]),
        np.diag(value_metric["matrix"]),
    ))
    chosen = np.argsort(sensitivity)[::-1][:total]
    return chosen[chosen < sensitivity.size // 2], (
        chosen[chosen >= sensitivity.size // 2] - sensitivity.size // 2
    )


def weighted_error(candidate, reference, metric):
    difference = candidate - reference
    numerator = np.einsum("ti,ij,tj->", difference, metric, difference)
    denominator = np.einsum("ti,ij,tj->", reference, metric, reference)
    return math.sqrt(max(numerator, 0.0) / max(denominator, 1e-30))


def vector_weighted_error(candidate, reference, metric):
    difference = candidate - reference
    numerator = matmul(matmul(difference, metric), difference)
    denominator = matmul(matmul(reference, metric), reference)
    return math.sqrt(max(numerator, 0.0) / max(denominator, 1e-30))


def fixed_basis_layout(rows, dimension, key_rank, value_rank):
    raw = 4 + 2 * (rows * dimension // 4 + rows * 2)
    raw += rows * (key_rank + value_rank) // 2
    raw += 32 * int(key_rank > 0) + 32 * int(value_rank > 0)
    addressed = (raw + 63) // 64 * 64 + 4
    return addressed, addressed * 8 / (2 * rows * dimension)


def build_variants(
    keys, values, metrics, rotation, centers2, centers3, centers4
):
    names = (
        "K4/V2-Q8",
        "Euclidean-TSVD-Q4",
        "Euclidean-eOpt-Q4",
        "Weighted-TSVD-Q4",
        "Weighted-TSVD-exact-factor",
        "Fixed-basis-Q4",
        "WH-upgrade-joint48",
        "WH-upgrade-K48/V0",
        "WH-upgrade-K32/V16",
        "WH-upgrade-K24/V24",
        "WH-upgrade-joint47+mask",
        "WH-upgrade-joint46+header",
        "WH-upgrade-K46/V0+header",
        "WH-upgrade-K32/V14+header",
        "WH-upgrade-K24/V22+header",
        "WH-K3/V2",
        "WH-K3+K4x31/V2+mask",
        "WH-K3/V2+V4x15+mask",
    )
    variants = {}
    ranks = {}
    recon = defaultdict(lambda: {name: [] for name in names})
    metric_recon = defaultdict(lambda: {name: [] for name in names})

    for identity, (key_metric, value_metric) in metrics.items():
        layer, head = identity
        original_k = half(stack(keys, layer, head, TILE))
        original_v = half(stack(values, layer, head, TILE))
        estimate_k = eopt_shrink(original_k, "paper")
        estimate_v = eopt_shrink(original_v, "paper")
        key_rank = estimate_k["rank"]
        value_rank = estimate_v["rank"]
        ranks[identity] = key_rank, value_rank
        euclidean_k = truncated(original_k, key_rank)
        euclidean_v = truncated(original_v, value_rank)
        weighted_k = weighted_rank(original_k, key_metric, key_rank)
        weighted_v = weighted_rank(original_v, value_metric, value_rank)
        qk, qv, _, _ = encode_pair(original_k, original_v, 8)
        pair = {
            "K4/V2-Q8": (qk, qv),
            "Euclidean-TSVD-Q4": (
                encode_signal(
                    original_k, euclidean_k, key_rank, rotation, centers2
                ),
                encode_signal(
                    original_v, euclidean_v, value_rank, rotation, centers2
                ),
            ),
            "Euclidean-eOpt-Q4": (
                encode_signal(
                    original_k, estimate_k["signal"], key_rank,
                    rotation, centers2,
                ),
                encode_signal(
                    original_v, estimate_v["signal"], value_rank,
                    rotation, centers2,
                ),
            ),
            "Weighted-TSVD-Q4": (
                encode_signal(
                    original_k, weighted_k, key_rank, rotation, centers2
                ),
                encode_signal(
                    original_v, weighted_v, value_rank, rotation, centers2
                ),
            ),
            "Weighted-TSVD-exact-factor": (
                tq_mse(
                    original_k - weighted_k, rotation, centers2
                )[0] + weighted_k,
                tq_mse(
                    original_v - weighted_v, rotation, centers2
                )[0] + weighted_v,
            ),
            "Fixed-basis-Q4": (
                quantize_fixed_basis(
                    original_k, key_metric, key_rank, rotation, centers2
                ),
                quantize_fixed_basis(
                    original_v, value_metric, value_rank, rotation, centers2
                ),
            ),
        }
        selections = {
            "WH-upgrade-joint48": jointly_selected(
                key_metric, value_metric, 48
            ),
            "WH-upgrade-K48/V0": selected_coordinates(
                key_metric, value_metric, 48, 0
            ),
            "WH-upgrade-K32/V16": selected_coordinates(
                key_metric, value_metric, 32, 16
            ),
            "WH-upgrade-K24/V24": selected_coordinates(
                key_metric, value_metric, 24, 24
            ),
            "WH-upgrade-joint47+mask": jointly_selected(
                key_metric, value_metric, 47
            ),
            "WH-upgrade-joint46+header": jointly_selected(
                key_metric, value_metric, 46
            ),
            "WH-upgrade-K46/V0+header": selected_coordinates(
                key_metric, value_metric, 46, 0
            ),
            "WH-upgrade-K32/V14+header": selected_coordinates(
                key_metric, value_metric, 32, 14
            ),
            "WH-upgrade-K24/V22+header": selected_coordinates(
                key_metric, value_metric, 24, 22
            ),
        }
        for name, (key_selected, value_selected) in selections.items():
            pair[name] = (
                mixed_precision(original_k, key_selected, centers2, centers4),
                mixed_precision(original_v, value_selected, centers2, centers4),
            )
        key31, _ = selected_coordinates(key_metric, value_metric, 31, 0)
        _, value15 = selected_coordinates(key_metric, value_metric, 0, 15)
        pair["WH-K3/V2"] = (
            mixed_precision(original_k, (), centers3, centers4),
            mixed_precision(original_v, (), centers2, centers4),
        )
        pair["WH-K3+K4x31/V2+mask"] = (
            mixed_precision(original_k, key31, centers3, centers4),
            mixed_precision(original_v, (), centers2, centers4),
        )
        pair["WH-K3/V2+V4x15+mask"] = (
            mixed_precision(original_k, (), centers3, centers4),
            mixed_precision(original_v, value15, centers2, centers4),
        )
        variants[identity] = pair

        for name, (candidate_k, candidate_v) in pair.items():
            recon["K"][name].append(relative_error(candidate_k, original_k))
            recon["V"][name].append(relative_error(candidate_v, original_v))
            metric_recon["K"][name].append(weighted_error(
                candidate_k, original_k, key_metric["matrix"]
            ))
            metric_recon["V"][name].append(weighted_error(
                candidate_v, original_v, value_metric["matrix"]
            ))
    return names, variants, ranks, recon, metric_recon


def evaluate_causal(keys, values, queries, metrics, names, variants):
    errors = {name: [] for name in names}
    score_errors = {name: [] for name in names}
    output_errors = {name: [] for name in names}
    grouped = defaultdict(list)
    for record in queries:
        if record[2] > TILE:
            grouped[record[0], record[1], record[2]].append(record)

    full = {}
    for layer, head in metrics:
        full[layer, head] = (
            half(stack(keys, layer, head, max(record[2] for record in queries))),
            half(stack(values, layer, head, max(record[2] for record in queries))),
        )

    for (layer, head, count), records in grouped.items():
        original_k, original_v = full[layer, head]
        exact_k = original_k[:count]
        exact_v = original_v[:count]
        queries_matrix = np.stack([record[6] for record in records]).astype(
            np.float64
        )
        scale = records[0][4]
        exact_scores = matmul(exact_k, queries_matrix.T) * scale
        exact_weights = np.exp(exact_scores - np.max(exact_scores, axis=0))
        exact_weights /= np.sum(exact_weights, axis=0)
        exact_output = matmul(exact_weights.T, exact_v)
        value_metric = metrics[layer, head][1]["matrix"]

        for name in names:
            tile_k, tile_v = variants[layer, head][name]
            candidate_scores = exact_scores.copy()
            candidate_scores[:TILE] = matmul(tile_k, queries_matrix.T) * scale
            candidate_weights = np.exp(
                candidate_scores - np.max(candidate_scores, axis=0)
            )
            candidate_weights /= np.sum(candidate_weights, axis=0)
            candidate_output = (
                matmul(candidate_weights[:TILE].T, tile_v)
                + matmul(candidate_weights[TILE:].T, exact_v[TILE:])
            )
            for index in range(len(records)):
                errors[name].append(relative_error(
                    candidate_output[index], exact_output[index]
                ))
                score_errors[name].append(relative_error(
                    candidate_scores[:, index], exact_scores[:, index]
                ))
                output_errors[name].append(vector_weighted_error(
                    candidate_output[index], exact_output[index], value_metric
                ))
    return errors, score_errors, output_errors


def evaluate_final_k3(
    keys, values, queries, metrics, centers2, centers3, centers4
):
    names = ("K4/V2-Q8", "WH-K3+K4x31/V2+mask")
    errors = {name: [] for name in names}
    score_errors = {name: [] for name in names}
    output_errors = {name: [] for name in names}
    grouped = defaultdict(list)
    for record in queries:
        if record[2] == 2 * TILE:
            grouped[record[0], record[1]].append(record)

    for (layer, head), records in grouped.items():
        original_k = half(stack(keys, layer, head, 2 * TILE))
        original_v = half(stack(values, layer, head, 2 * TILE))
        key_metric, value_metric = metrics[layer, head]
        key31, _ = selected_coordinates(key_metric, value_metric, 31, 0)
        k4_key = np.empty_like(original_k)
        k4_value = np.empty_like(original_v)
        for start in range(0, 2 * TILE, TILE):
            region = slice(start, start + TILE)
            k4_key[region], k4_value[region], _, _ = encode_pair(
                original_k[region], original_v[region], 8
            )
        candidates = {
            "K4/V2-Q8": (k4_key, k4_value),
            "WH-K3+K4x31/V2+mask": (
                mixed_precision(original_k, key31, centers3, centers4),
                mixed_precision(original_v, (), centers2, centers4),
            ),
        }
        query_matrix = np.stack([record[6] for record in records]).astype(
            np.float64
        )
        scale = records[0][4]
        exact_scores = matmul(original_k, query_matrix.T) * scale
        exact_weights = np.exp(exact_scores - np.max(exact_scores, axis=0))
        exact_weights /= np.sum(exact_weights, axis=0)
        exact_output = matmul(exact_weights.T, original_v)
        for name, (candidate_k, candidate_v) in candidates.items():
            candidate_scores = matmul(candidate_k, query_matrix.T) * scale
            candidate_weights = np.exp(
                candidate_scores - np.max(candidate_scores, axis=0)
            )
            candidate_weights /= np.sum(candidate_weights, axis=0)
            candidate_output = matmul(candidate_weights.T, candidate_v)
            for index in range(len(records)):
                errors[name].append(relative_error(
                    candidate_output[index], exact_output[index]
                ))
                score_errors[name].append(relative_error(
                    candidate_scores[:, index], exact_scores[:, index]
                ))
                output_errors[name].append(vector_weighted_error(
                    candidate_output[index], exact_output[index],
                    value_metric["matrix"],
                ))
    return errors, score_errors, output_errors


def print_metric_summary(metrics):
    for label, index in (("K", 0), ("V", 1)):
        conditions = np.asarray([
            pair[index]["condition"] for pair in metrics.values()
        ])
        dropped = np.asarray([
            pair[index]["dropped"] for pair in metrics.values()
        ])
        print(
            f"metric {label}: condition mean={np.mean(conditions):.3e} "
            f"p95={np.percentile(conditions, 95):.3e} "
            f"max={np.max(conditions):.3e}; "
            f"support drops total={np.sum(dropped)} max={np.max(dropped)} "
            f"rcond={SUPPORT_RCOND:g}"
        )


def print_results(names, errors, score_errors, output_errors, recon, metric_recon):
    for name in names:
        print(f"{name:26s} attention {summarize(errors[name])}")
        print(f"{'':26s} K-score  {summarize(score_errors[name])}")
        print(f"{'':26s} post-WO  {summarize(output_errors[name])}")
        print(f"{'':26s} K-recon  {summarize(recon['K'][name])}")
        print(f"{'':26s} V-recon  {summarize(recon['V'][name])}")
        print(f"{'':26s} K-metric {summarize(metric_recon['K'][name])}")
        print(f"{'':26s} V-metric {summarize(metric_recon['V'][name])}")


def print_final_results(result):
    errors, score_errors, output_errors = result
    for name in errors:
        print(f"{name:26s} attention {summarize(errors[name])}")
        print(f"{'':26s} K-score  {summarize(score_errors[name])}")
        print(f"{'':26s} post-WO  {summarize(output_errors[name])}")


def print_storage(ranks, dimension, identity_count):
    layouts = [pair_layout(TILE, dimension, *rank) for rank in ranks.values()]
    spectral = np.asarray([layout["counted"] for layout in layouts])
    spectral_bytes = np.asarray([
        layout["addressed_bytes"] for layout in layouts
    ])
    fixed = np.asarray([
        fixed_basis_layout(TILE, dimension, *rank)[1]
        for rank in ranks.values()
    ])
    rank_sum = np.asarray([sum(rank) for rank in ranks.values()])
    print(
        f"rank-matched rK+rV mean={np.mean(rank_sum):.3f} "
        f"p95={np.percentile(rank_sum, 95):.1f} max={np.max(rank_sum)}"
    )
    print(
        f"spectral Q4 record mean={np.mean(spectral):.6f} "
        f"p95={np.percentile(spectral, 95):.6f} max={np.max(spectral):.6f} "
        f"pass<=3={100 * np.mean(spectral <= 3):.2f}%"
    )
    print(
        f"fixed-basis Q4 record mean={np.mean(fixed):.6f} "
        f"p95={np.percentile(fixed, 95):.6f} max={np.max(fixed):.6f}"
    )
    print("structured 128-token pair records including masks and alignment:")
    for name, raw in (
        ("Q2/Q4 joint48", 6160),
        ("Q2/Q4 joint47", 6128),
        ("Q2/Q4 rank46+header", 6100),
        ("K3+K4x31/V2", 6136),
        ("K3/V2+V4x15", 6120),
    ):
        addressed = (raw + 63) // 64 * 64
        print(
            f"  {name:18s} raw={raw} B aligned={addressed} B "
            f"rate={addressed * 8 / (2 * TILE * dimension):.6f}"
        )
    print(
        "rank46 body is 380 bits/token; an immediate partial-tail pack has "
        "no FP16 tail, while its one-time masks/header still need lifecycle "
        "accounting below a complete tile"
    )

    spectral_decoder = (dimension * dimension + 4) * 4
    metric_state = 2 * (dimension * dimension * 2 + dimension * 4)
    fixed_basis_state = np.mean(rank_sum) * dimension * 2
    for context in (4095, 16383, 65535):
        sealed, tail = divmod(context, TILE)
        common = (
            sealed * np.mean(spectral_bytes)
            + tail * 2 * dimension * 2
            + spectral_decoder / identity_count
        )
        denominator = 2 * context * dimension
        euclidean_rate = common * 8 / denominator
        weighted_rate = (common + metric_state) * 8 / denominator
        fixed_rate = (
            sealed * np.mean([
                fixed_basis_layout(TILE, dimension, *rank)[0]
                for rank in ranks.values()
            ])
            + tail * 2 * dimension * 2
            + spectral_decoder / identity_count
            + fixed_basis_state
        ) * 8 / denominator
        print(
            f"context={context:5d} spectral+tail={euclidean_rate:.6f} "
            f"weighted+metric-state={weighted_rate:.6f} "
            f"fixed-basis+state={fixed_rate:.6f}"
        )
    shared_decoder = (dimension * dimension + 4 + 8 + 16) * 4
    print(
        f"comparison harness constants={shared_decoder} bytes over "
        f"{identity_count} layer/head identities "
        f"(dense Haar plus analytic Q2/Q3/Q4 levels)"
    )


def self_check():
    generator = np.random.Generator(np.random.PCG64(7))
    source = generator.standard_normal((4, 4))
    metric = matmul(source, source.T)
    direct = np.zeros_like(metric)
    for position in range(5):
        angles = position * 10000.0 ** (-2 * np.arange(2) / 4)
        cosine = np.cos(angles)
        sine = np.sin(angles)
        rope = np.zeros((4, 4))
        for index in range(2):
            rope[index, index] = cosine[index]
            rope[index, index + 2] = -sine[index]
            rope[index + 2, index] = sine[index]
            rope[index + 2, index + 2] = cosine[index]
        direct += matmul(matmul(rope, metric), rope.T) / 5
    analytic = rope_average(metric, 4, 10000.0, 5)
    if not np.allclose(analytic, direct, rtol=1e-12, atol=1e-12):
        raise RuntimeError("RoPE covariance average self-check failed")
    normalized = psd_metric(metric)
    matrix = generator.standard_normal((7, 4))
    recovered = weighted_rank(matrix, normalized, 4)
    if not np.allclose(recovered, matrix, rtol=1e-10, atol=1e-10):
        raise RuntimeError("weighted full-rank reconstruction self-check failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="all-token D64 LaplaceKV trace")
    parser.add_argument("model", help="matching Qwen2.5 GGUF")
    parser.add_argument("--rope-horizon", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=260502905)
    args = parser.parse_args()

    self_check()
    keys, values, queries = load_trace(args.trace)
    if not queries or queries[0][6].size != 64:
        raise ValueError("requires a D64 trace")
    dimension = queries[0][6].size
    layers = sorted({record[0] for record in queries})
    reader = gguf.GGUFReader(args.model, "r")
    tensors = tensor_map(reader)
    metrics, geometry = build_metrics(
        reader, tensors, layers, dimension, args.rope_horizon
    )

    generator = np.random.Generator(np.random.PCG64(args.seed))
    random_matrix = generator.standard_normal((dimension, dimension))
    rotation, upper = np.linalg.qr(random_matrix)
    rotation *= np.where(np.diag(upper) < 0, -1.0, 1.0)
    centers2 = gaussian_lloyd(2, dimension)
    centers3 = gaussian_lloyd(3, dimension)
    centers4 = gaussian_lloyd(4, dimension)

    print(
        f"trace={args.trace} model={args.model} layers={len(layers)} "
        f"D={dimension} GQA={geometry['q_heads']}Q/{geometry['kv_heads']}KV "
        f"rope_base={geometry['rope_base']:g} "
        f"absolute_prior=0..{args.rope_horizon - 1} trace_domain=WH "
        f"weight_types={geometry['weight_types']}"
    )
    print_metric_summary(metrics)
    print(
        "joint K/V coordinate selection uses separately trace-normalized "
        "metrics, so it is an equal-total proxy rather than raw physical loss"
    )
    names, variants, ranks, recon, metric_recon = build_variants(
        keys, values, metrics, rotation, centers2, centers3, centers4
    )
    errors, score_errors, output_errors = evaluate_causal(
        keys, values, queries, metrics, names, variants
    )
    print("\n[causal first tile sealed at 128, queries 129..256]")
    print_results(
        names, errors, score_errors, output_errors, recon, metric_recon
    )
    print("\n[current 256-token prefix, both tiles encoded]")
    print_final_results(evaluate_final_k3(
        keys, values, queries, metrics, centers2, centers3, centers4
    ))
    print("\n[fully counted storage]")
    print_storage(ranks, dimension, len(metrics))


if __name__ == "__main__":
    main()

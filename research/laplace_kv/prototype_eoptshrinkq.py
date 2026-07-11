#!/usr/bin/env python3
"""Reproduce the public eOptShrinkQ construction on Laplace KV traces.

This is a research harness. It does not change the inference engine.
"""

import argparse
import math
from statistics import NormalDist

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack
from prototype_syndrome_embedding import encode_pair


BLOCK = 128
RESIDUAL_BITS = 2
FACTOR_BITS = 4
Q8_METADATA_BITS = 4800
PAIR_HEADER_BYTES = 4
PAIR_OFFSET_BYTES = 4
ALIGNMENT = 64


def half(values):
    return np.asarray(values).astype(np.float16).astype(np.float32)


def align(value, multiple=ALIGNMENT):
    return (value + multiple - 1) // multiple * multiple


def gaussian_lloyd(bits, dimension, iterations=256):
    """Lloyd-Max centroids for N(0, 1/d), computed analytically."""
    count = 1 << bits
    normal = NormalDist()
    centers = np.asarray([
        normal.inv_cdf((index + 0.5) / count)
        for index in range(count)
    ], dtype=np.float64)
    root_two_pi = math.sqrt(2.0 * math.pi)

    def density(value):
        return math.exp(-0.5 * value * value) / root_two_pi

    def cdf(value):
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    for _ in range(iterations):
        edges = (centers[:-1] + centers[1:]) * 0.5
        updated = np.empty_like(centers)
        for index in range(count):
            low = -math.inf if index == 0 else float(edges[index - 1])
            high = math.inf if index == count - 1 else float(edges[index])
            probability = cdf(high) - cdf(low)
            low_density = 0.0 if math.isinf(low) else density(low)
            high_density = 0.0 if math.isinf(high) else density(high)
            updated[index] = (low_density - high_density) / probability
        if np.max(np.abs(updated - centers)) < 1e-14:
            centers = updated
            break
        centers = updated
    return centers / math.sqrt(dimension)


def haar_rotation(dimension, seed):
    """One deterministic representative of the unspecified shared Haar draw."""
    generator = np.random.Generator(np.random.PCG64(seed))
    matrix = generator.standard_normal((dimension, dimension))
    orthogonal, upper = np.linalg.qr(matrix)
    signs = np.sign(np.diag(upper))
    signs[signs == 0] = 1
    return orthogonal * signs


def nearest_indices(values, centers):
    flat = np.asarray(values, dtype=np.float64).ravel()
    indices = np.argmin(np.abs(flat[:, None] - centers[None, :]), axis=1)
    return indices.reshape(np.asarray(values).shape)


def matmul(left, right):
    # NumPy 2.0's Accelerate wrapper can report floating-point status flags
    # raised inside BLAS even when the finite matrix product is valid.
    with np.errstate(all="ignore"):
        output = np.matmul(left, right)
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("non-finite matrix product")
    return output


def adaptive_lloyd(samples, bits, iterations=64):
    samples = np.asarray(samples, dtype=np.float64).ravel()
    count = 1 << bits
    centers = np.quantile(samples, (np.arange(count) + 0.5) / count)
    for _ in range(iterations):
        labels = nearest_indices(samples, centers).ravel()
        updated = centers.copy()
        for index in range(count):
            selected = samples[labels == index]
            if selected.size:
                updated[index] = np.mean(selected)
        updated.sort()
        if np.max(np.abs(updated - centers)) < 1e-12:
            centers = updated
            break
        centers = updated
    return half(centers).astype(np.float64)


def tq_mse(matrix, rotation, centers):
    """Algorithm 1 TQ_MSE, including the stored FP16 norm per row."""
    source = np.asarray(matrix, dtype=np.float64)
    exact_norms = np.linalg.norm(source, axis=1)
    stored_norms = half(exact_norms).astype(np.float64)
    unit = np.zeros_like(source)
    nonzero = exact_norms > 0
    unit[nonzero] = source[nonzero] / exact_norms[nonzero, None]
    rotated = matmul(unit, rotation.T)
    codes = nearest_indices(rotated, centers)
    reconstructed = matmul(centers[codes], rotation)
    reconstructed *= stored_norms[:, None]
    return reconstructed, codes, stored_norms


def eopt_shrink(matrix, imputation):
    """Algorithms 2/3 plus D-transform details from cited eOptShrink."""
    source = np.asarray(matrix, dtype=np.float64)
    rows, dimension = source.shape
    left, singular, right = np.linalg.svd(source, full_matrices=False)
    eigenvalues = singular * singular
    spectral_size = eigenvalues.size
    longer = max(rows, dimension)
    shorter = min(rows, dimension)
    beta = shorter / longer

    c = min(1.0 / 2.01, 1.0 / math.log(math.log(dimension)))
    k = int(math.floor(dimension ** c))
    if 2 * k >= spectral_size:
        raise ValueError("eOptShrink pilot exceeds the observed spectrum")
    denominator = 2.0 ** (2.0 / 3.0) - 1.0
    edge = (
        eigenvalues[k]
        + (eigenvalues[k] - eigenvalues[2 * k]) / denominator
    )
    threshold = dimension ** (-1.0 / 3.0)
    rank = int(np.count_nonzero(eigenvalues / edge - 1.0 > threshold))
    if rank + 2 * k >= spectral_size:
        raise ValueError(
            f"eOptShrink cannot impute rank={rank} with k={k} "
            f"from a spectrum of length {spectral_size}"
        )

    if rank == 0:
        return {
            "signal": np.zeros_like(source),
            "left": left[:, :0],
            "right": right[:0],
            "shrunk": singular[:0],
            "singular": singular,
            "rank": 0,
            "k": k,
            "edge": edge,
        }

    anchor = eigenvalues[rank + k]
    second = eigenvalues[rank + 2 * k]
    imputed = np.empty(k, dtype=np.float64)
    for local in range(k):
        # Algorithm 2 in arXiv:2605.02905 uses j=1,...,k. The cited
        # eOptShrink paper instead uses a zero-based local exponent.
        numerator = local + 1 if imputation == "paper" else local
        fraction = (numerator / k) ** (2.0 / 3.0)
        imputed[local] = anchor + (1.0 - fraction) * (
            anchor - second
        ) / denominator
    noise = np.concatenate((imputed, eigenvalues[rank + k:]))
    if noise.size != spectral_size - rank:
        raise RuntimeError("estimated noise spectrum has the wrong length")

    shrunk = np.empty(rank, dtype=np.float64)
    for index in range(rank):
        observed = eigenvalues[index]
        gaps = noise - observed
        m1 = np.mean(1.0 / gaps)
        m1_prime = np.mean(1.0 / (gaps * gaps))
        # This minus sign is required by m(z)=int (x-z)^-1 dF(x), and
        # matches the cited author's MATLAB. The displayed cited-paper
        # equation has a conflicting plus sign.
        m2 = -(1.0 - beta) / observed + beta * m1
        m2_prime = (1.0 - beta) / (observed * observed) + beta * m1_prime
        transform = observed * m1 * m2
        derivative = (
            m1 * m2
            + observed * m1_prime * m2
            + observed * m1 * m2_prime
        )
        strength = 1.0 / math.sqrt(transform)
        overlap1 = m1 / (strength * strength * derivative)
        overlap2 = m2 / (strength * strength * derivative)
        if overlap1 <= 0 or overlap2 <= 0:
            raise ValueError(
                "eOptShrink produced a non-positive singular-vector overlap"
            )
        shrunk[index] = strength * math.sqrt(overlap1 * overlap2)

    signal = matmul(left[:, :rank] * shrunk, right[:rank])
    return {
        "signal": signal,
        "left": left[:, :rank],
        "right": right[:rank],
        "shrunk": shrunk,
        "singular": singular,
        "rank": rank,
        "k": k,
        "edge": edge,
    }


def quantized_factors(left, weights, right):
    rank = len(weights)
    if rank == 0:
        return np.zeros((left.shape[0], right.shape[1]), dtype=np.float64)
    left = np.asarray(left, dtype=np.float64).copy()
    right = np.asarray(right, dtype=np.float64).copy()
    for component in range(rank):
        pivot = np.argmax(np.abs(left[:, component]))
        if left[pivot, component] < 0:
            left[:, component] *= -1
            right[component] *= -1
    samples = np.concatenate((left.ravel(), right.ravel()))
    codebook = adaptive_lloyd(samples, FACTOR_BITS)
    left_codes = nearest_indices(left, codebook)
    right_codes = nearest_indices(right, codebook)
    stored_weights = half(weights).astype(np.float64)
    return matmul(
        codebook[left_codes] * stored_weights, codebook[right_codes]
    )


def encode_spectral(matrix, rotation, centers, imputation):
    estimate = eopt_shrink(matrix, imputation)
    residual = np.asarray(matrix, dtype=np.float64) - estimate["signal"]
    decoded_residual, _, _ = tq_mse(residual, rotation, centers)
    exact_factor = estimate["signal"] + decoded_residual
    quantized_factor = quantized_factors(
        estimate["left"], estimate["shrunk"], estimate["right"]
    ) + decoded_residual

    rank = estimate["rank"]
    truncated_signal = matmul(
        estimate["left"] * estimate["singular"][:rank],
        estimate["right"],
    )
    truncated_residual = np.asarray(matrix, dtype=np.float64) - truncated_signal
    decoded_truncated, _, _ = tq_mse(
        truncated_residual, rotation, centers
    )
    quantized_truncated = quantized_factors(
        estimate["left"], estimate["singular"][:rank], estimate["right"]
    ) + decoded_truncated
    return quantized_factor, exact_factor, quantized_truncated, estimate


def attention(query, scale, keys, values):
    scores = np.einsum(
        "td,d->t", keys.astype(np.float64), query.astype(np.float64)
    ) * scale
    weights = softmax(scores)
    return np.einsum("t,td->d", weights, values.astype(np.float64))


def summarize(values):
    percent = np.asarray(values, dtype=np.float64) * 100.0
    return (
        f"mean={np.mean(percent):8.3f}% "
        f"p95={np.percentile(percent, 95):8.3f}% "
        f"max={np.max(percent):8.3f}%"
    )


def matrix_raw_bytes(rows, dimension, rank, split_codebooks=False):
    residual_codes = (rows * dimension * RESIDUAL_BITS + 7) // 8
    norms = rows * 2
    factor_codes = (rank * (rows + dimension) * FACTOR_BITS + 7) // 8
    singular = rank * 2
    codebooks = (64 if split_codebooks else 32) if rank else 0
    return residual_codes + norms + factor_codes + singular + codebooks


def pair_layout(rows, dimension, key_rank, value_rank, split_codebooks=False):
    raw = PAIR_HEADER_BYTES
    raw += matrix_raw_bytes(
        rows, dimension, key_rank, split_codebooks=split_codebooks
    )
    raw += matrix_raw_bytes(
        rows, dimension, value_rank, split_codebooks=split_codebooks
    )
    addressed = align(raw) + PAIR_OFFSET_BYTES
    split_raw = PAIR_HEADER_BYTES
    split_raw += matrix_raw_bytes(
        rows, dimension, key_rank, split_codebooks=True
    )
    split_raw += matrix_raw_bytes(
        rows, dimension, value_rank, split_codebooks=True
    )
    split_addressed = align(split_raw) + PAIR_OFFSET_BYTES
    scalars = 2 * rows * dimension
    paper_bits = (
        2 * rows * dimension * RESIDUAL_BITS
        + (key_rank + value_rank)
        * (rows + dimension) * FACTOR_BITS
    )
    disclosed_bits = paper_bits + 2 * rows * 16 + (
        key_rank + value_rank
    ) * 16
    return {
        "paper": paper_bits / scalars,
        "disclosed": disclosed_bits / scalars,
        "raw_bytes": raw,
        "addressed_bytes": addressed,
        "counted": addressed * 8 / scalars,
        "split_bytes": split_addressed,
        "counted_split": split_addressed * 8 / scalars,
    }


def k4v2_q8_layout(rows, dimension):
    body_bits = rows * dimension * 6
    raw = (body_bits + Q8_METADATA_BITS + 7) // 8
    return raw, align(raw) + PAIR_OFFSET_BYTES


def evaluate(keys, values, queries, start, rotation, centers, imputation):
    names = (
        "K4/V2-Q8",
        "TQ2",
        "SVD-rankmatched+TQ2",
        "eOptShrinkQ-Q4factors",
        "eOptShrinkQ-exact-factors",
    )
    errors = {name: [] for name in names}
    score_errors = {name: [] for name in names}
    key_errors = {name: [] for name in names}
    value_errors = {name: [] for name in names}
    ranks = []
    layouts = []
    cache = {}

    for layer, head, count, first, scale, _, query, _ in queries:
        identity = layer, head, count, first
        if identity not in cache:
            original_k = half(stack(keys, layer, head, count)[first:])
            original_v = half(stack(values, layer, head, count)[first:])
            variants_k = {name: original_k.copy() for name in names}
            variants_v = {name: original_v.copy() for name in names}

            for tile_start in range(start, len(original_k), BLOCK):
                tile_k = original_k[tile_start:tile_start + BLOCK]
                tile_v = original_v[tile_start:tile_start + BLOCK]
                if len(tile_k) != BLOCK:
                    raise ValueError("trace ends in a partial spectral block")

                qk, qv, _, _ = encode_pair(tile_k, tile_v, 8)
                variants_k["K4/V2-Q8"][tile_start:tile_start + BLOCK] = qk
                variants_v["K4/V2-Q8"][tile_start:tile_start + BLOCK] = qv

                tqk, _, _ = tq_mse(tile_k, rotation, centers)
                tqv, _, _ = tq_mse(tile_v, rotation, centers)
                variants_k["TQ2"][tile_start:tile_start + BLOCK] = tqk
                variants_v["TQ2"][tile_start:tile_start + BLOCK] = tqv

                e_k, exact_k, svd_k, estimate_k = encode_spectral(
                    tile_k, rotation, centers, imputation
                )
                e_v, exact_v, svd_v, estimate_v = encode_spectral(
                    tile_v, rotation, centers, imputation
                )
                variants_k["SVD-rankmatched+TQ2"][tile_start:tile_start + BLOCK] = svd_k
                variants_v["SVD-rankmatched+TQ2"][tile_start:tile_start + BLOCK] = svd_v
                variants_k["eOptShrinkQ-Q4factors"][tile_start:tile_start + BLOCK] = e_k
                variants_v["eOptShrinkQ-Q4factors"][tile_start:tile_start + BLOCK] = e_v
                variants_k["eOptShrinkQ-exact-factors"][tile_start:tile_start + BLOCK] = exact_k
                variants_v["eOptShrinkQ-exact-factors"][tile_start:tile_start + BLOCK] = exact_v

                key_rank = estimate_k["rank"]
                value_rank = estimate_v["rank"]
                ranks.append((key_rank, value_rank))
                layouts.append(pair_layout(BLOCK, tile_k.shape[1], key_rank, value_rank))
                for name in names:
                    key_errors[name].append(relative_error(
                        variants_k[name][tile_start:tile_start + BLOCK], tile_k
                    ))
                    value_errors[name].append(relative_error(
                        variants_v[name][tile_start:tile_start + BLOCK], tile_v
                    ))
            cache[identity] = original_k, original_v, variants_k, variants_v

        original_k, original_v, variants_k, variants_v = cache[identity]
        exact_scores = np.einsum(
            "td,d->t",
            original_k.astype(np.float64),
            query.astype(np.float64),
        ) * scale
        exact = attention(query, scale, original_k, original_v)
        for name in names:
            candidate_scores = np.einsum(
                "td,d->t",
                variants_k[name].astype(np.float64),
                query.astype(np.float64),
            ) * scale
            score_errors[name].append(
                relative_error(candidate_scores, exact_scores)
            )
            candidate = attention(
                query, scale, variants_k[name], variants_v[name]
            )
            errors[name].append(relative_error(candidate, exact))

    return errors, score_errors, key_errors, value_errors, ranks, layouts


def print_layouts(layouts, ranks, tiles, dimension):
    key_ranks = np.asarray([item[0] for item in ranks])
    value_ranks = np.asarray([item[1] for item in ranks])
    print(
        f"ranks K mean={np.mean(key_ranks):.3f} p95={np.percentile(key_ranks, 95):.1f} "
        f"max={np.max(key_ranks)}; V mean={np.mean(value_ranks):.3f} "
        f"p95={np.percentile(value_ranks, 95):.1f} max={np.max(value_ranks)}"
    )
    for field in ("paper", "disclosed", "counted", "counted_split"):
        values = np.asarray([layout[field] for layout in layouts])
        print(
            f"storage {field:9s} mean={np.mean(values):.6f} "
            f"p95={np.percentile(values, 95):.6f} max={np.max(values):.6f} "
            f"pass<=3={np.mean(values <= 3.0) * 100:.2f}%"
        )
    raw = np.asarray([layout["raw_bytes"] for layout in layouts])
    addressed = np.asarray([layout["addressed_bytes"] for layout in layouts])
    print(
        f"record bytes raw mean={np.mean(raw):.2f} max={np.max(raw)}; "
        f"aligned+offset mean={np.mean(addressed):.2f} max={np.max(addressed)}"
    )

    rotation_bytes = dimension * dimension * 4
    tq_table_bytes = (1 << RESIDUAL_BITS) * 4
    decoder_bytes = rotation_bytes + tq_table_bytes
    scalar_count = tiles * 2 * BLOCK * dimension
    decoder_rate = decoder_bytes * 8 / scalar_count
    counted_mean = np.mean([layout["counted"] for layout in layouts])
    print(
        f"shared decoder table fp32={decoder_bytes} bytes "
        f"({decoder_rate:.6f} bits/scalar over {tiles} tile pairs); "
        f"counted+table mean={counted_mean + decoder_rate:.6f}"
    )
    print(
        f"prototype table fp64={dimension * dimension * 8 + (1 << RESIDUAL_BITS) * 8} bytes"
    )


def print_traffic(ranks, dimension):
    rank_sum = np.mean([key + value for key, value in ranks])
    addressed = np.mean([
        pair_layout(BLOCK, dimension, key, value)["addressed_bytes"]
        for key, value in ranks
    ])
    fp16_bytes = 2 * BLOCK * dimension * 2
    k4_raw, k4_addressed = k4v2_q8_layout(BLOCK, dimension)
    factor_fmas = rank_sum * (BLOCK + dimension)
    residual_fmas = 2 * BLOCK * dimension
    print("direct-factor Apple attention model per mean 128-token tile pair:")
    print(
        f"  traffic eOpt={addressed:.2f} B FP16={fp16_bytes} B "
        f"K4/V2-Q8={k4_addressed} B (raw {k4_raw} B)"
    )
    print(
        f"  eOpt cache FMAs ~= residual {residual_fmas:.0f} + factors "
        f"{factor_fmas:.0f}; shared per-head transforms add "
        f"{2 * dimension * dimension} FMAs once per decode step"
    )
    for context in (16384, 65536):
        tile_count = context // BLOCK
        eopt_traffic = addressed * tile_count
        fp16_traffic = 4 * context * dimension
        eopt_fmas = (
            2 * context * dimension
            + factor_fmas * tile_count
            + 2 * dimension * dimension
        )
        fp16_fmas = 2 * context * dimension
        print(
            f"  T={context}: traffic={eopt_traffic / 1048576:.3f} MiB "
            f"vs FP16={fp16_traffic / 1048576:.3f} MiB "
            f"({fp16_traffic / eopt_traffic:.2f}x less); "
            f"arithmetic={eopt_fmas / fp16_fmas:.3f}x FP16"
        )

    tail_bytes = (BLOCK - 1) * 2 * dimension * 2
    mutable_state = (
        BLOCK * 4
        + 3 * dimension * 4
        + math.ceil(rank_sum / 2) * 4
    )
    print(
        f"mutable FP16 tail worst case={tail_bytes} bytes; "
        f"decode scratch lower bound~={mutable_state} bytes "
        f"(weights + rotated query/accumulator/output + factor temporaries)"
    )
    decoder_bytes = dimension * dimension * 4 + (1 << RESIDUAL_BITS) * 4
    for context in (4095, 16383, 65535):
        sealed = context // BLOCK
        tail = context % BLOCK
        total = addressed * sealed + tail * 2 * dimension * 2 + decoder_bytes
        rate = total * 8 / (2 * context * dimension)
        print(
            f"  effective storage T={context} (tail={tail}): "
            f"{rate:.6f} bits/scalar including table, records, and FP16 tail"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument(
        "--imputation", choices=("paper", "cited"), default="paper"
    )
    parser.add_argument("--seed", type=int, default=260502905)
    args = parser.parse_args()

    keys, values, queries = load_trace(args.trace)
    if not queries:
        raise ValueError("trace contains no queries")
    dimension = queries[0][6].size
    if dimension != 64:
        raise ValueError("this first-stage reproduction requires a D64 trace")
    if any(item[6].size != dimension for item in queries):
        raise ValueError("trace mixes head dimensions")
    tokens = max(item[2] - item[3] for item in queries)
    if tokens % BLOCK:
        raise ValueError("trace context is not divisible by 128")
    queries = [item for item in queries if item[2] - item[3] == tokens]

    rotation = haar_rotation(dimension, args.seed)
    centers = gaussian_lloyd(RESIDUAL_BITS, dimension)
    print(
        f"trace={args.trace} queries={len(queries)} tokens={tokens} dim={dimension} "
        f"block={BLOCK} residual=TQ_MSE-{RESIDUAL_BITS} "
        f"factor=adaptive-Lloyd-{FACTOR_BITS} imputation={args.imputation} "
        f"haar_seed={args.seed}"
    )
    print("TQ Gaussian centers:", " ".join(f"{value:.9f}" for value in centers))

    final = None
    for label, start in (("sealed-tile", BLOCK), ("full-prefix", 0)):
        print(f"\n[{label}] uncompressed_prefix={start}")
        result = evaluate(
            keys, values, queries, start, rotation, centers, args.imputation
        )
        errors, score_errors, key_errors, value_errors, ranks, layouts = result
        print(f"{'Exact FP16':29s} attention mean=   0.000% p95=   0.000% max=   0.000%")
        for name in errors:
            print(f"{name:29s} attention {summarize(errors[name])}")
            print(f"{'':29s} K-score  {summarize(score_errors[name])}")
            print(f"{'':29s} K-recon  {summarize(key_errors[name])}")
            print(f"{'':29s} V-recon  {summarize(value_errors[name])}")
        print_layouts(layouts, ranks, len(layouts), dimension)
        final = ranks

    print("\n[cost model, full-prefix rank distribution]")
    print_traffic(final, dimension)


if __name__ == "__main__":
    main()

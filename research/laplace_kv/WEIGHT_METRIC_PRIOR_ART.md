# Weight-derived metric prior-art audit

Date: 2026-07-12

## Decision

A generic "attention-aware weighted SVD plus direct factor attention" is not a
novel LaplaceKV foundation. Every broad component is already occupied:

- downstream-aware key/value error metrics;
- covariance-weighted or activation-weighted low-rank approximation;
- low-rank KV representations;
- direct attention in the compressed space;
- folding key factors into the query projection and value factors into the
  output projection; and
- spectral cache-block decomposition followed by residual quantization.

One narrower experiment remains plausibly distinct: derive fixed per-layer and
per-head quadratic metrics only from the frozen model weights and architecture,
then use those metrics inside a cache-block spectral shrinker and residual
quantizer. This means no calibration corpus, no prompt statistics, no online
basis update, and no model training. The literature reviewed here does not
show that complete construction.

That distinction is not enough for a novelty claim by itself. Expected
Attention already derives the query covariance from `W_Q`, hidden-state
covariance, and RoPE, and already measures value influence after `W_O`.
Replacing its hidden-state covariance with an identity or RMSNorm-gain proxy is
a calibration-free approximation to a published objective. A paper-worthy
Laplace contribution would need a new weighted spiked-model shrinker or rate
allocation rule, a proof or strong derivation for the frozen-weight proxy, and
a direct Apple Silicon representation that wins the measured storage, quality,
and latency Pareto frontier.

## Candidate objective

For row-vector attention, let a sealed key or value tile be `X` and let
`X_hat` be its compressed reconstruction. A right-side positive semidefinite
metric `M` gives the weighted loss

```text
L_M(X_hat) = ||(X - X_hat) M^(1/2)||_F^2.
```

For a key error `e_k`, expected squared logit error is

```text
E[(q e_k^T)^2] = e_k E[q^T q] e_k^T.
```

With `q = x W_Q`, a weight-only isotropic-input proxy is

```text
M_K = W_Q^T W_Q.
```

For GQA, `M_K` must sum or average the metric of every query head sharing the
KV head. For post-RoPE cached keys, the metric also depends on the relative
position distribution. A fixed method must average the RoPE-conjugated metric
analytically rather than silently ignoring position.

For a value error `e_v` in one head, the local post-output-projection error is

```text
||e_v W_O,h||_2^2 = e_v (W_O,h W_O,h^T) e_v^T,
```

so

```text
M_V = W_O,h W_O,h^T
```

is an exact coordinate metric conditional on the attention weight and on
holding the other token errors fixed. It is not a complete tile objective:
attention weights create cross-token terms, and key error changes the softmax
weights themselves.

The simple rank-`r` weighted approximation can be obtained from the truncated
SVD of `X M^(1/2)`. That alone is standard weighted low-rank approximation.
The unoccupied technical question is whether eOptShrink-style signal/noise
separation remains valid under the attention metric and whether its residual
can be coded below three fully counted bits while improving actual attention
error.

## Closest primary work

| Work | What it occupies | Difference from the narrow candidate |
|------|------------------|--------------------------------------|
| [OSCAR](https://arxiv.org/abs/2605.17757) | Attention-aware key and value covariance objectives, per-layer spectral rotations, INT2 cache, fused paged decode. Keys use empirical `Q^T Q`; values use an empirical score-aware covariance. | Requires a calibration corpus and uses rotations rather than a cache-tile low-rank-plus-residual shrinker. It does not use a frozen `W_O W_O^T` value metric as its compression objective. |
| [Expected Attention](https://arxiv.org/abs/2510.00636) | Derives future-query covariance as a RoPE transform of `W_Q Sigma_x W_Q^T`, and scores value influence with `||W_O v||`. | Uses online query statistics and evicts tokens. The weight-only metric is essentially its `Sigma_x = I` or fixed-normalization special case, so the objective itself is not a clean novelty claim. |
| [eOptShrinkQ](https://arxiv.org/abs/2605.02905) | Cache-block spiked model, automatic spectral rank selection, optimal Frobenius shrinkage, quantized low-rank factors, and TurboQuant residual coding below three displayed bits. | Its shrinker is Euclidean/Frobenius, not weighted by a frozen attention metric. Direct factor consumption is an implementation consequence, not a new Laplace claim. |
| [Palu](https://arxiv.org/abs/2407.21118) | SVD of `W_K` and `W_V`, caching low-dimensional states, folding the key reconstruction into `W_Q`, and folding the value reconstruction into `W_O`. | Decomposes projection weights, not each realized cache tile, and is not an attention-metric residual shrinker. It fully occupies factor folding. |
| [Eigen Attention](https://arxiv.org/abs/2408.05646) | SVD bases from calibration activations, low-rank KV attention, and basis fusion into `W_Q`, `W_K`, `W_V`, and `W_O`. | Static calibration basis rather than a frozen-weight metric or per-tile shrinker. It occupies compressed-space attention and model-weight fusion. |
| [OjaKV](https://aclanthology.org/2026.findings-acl.494/) | Online PCA bases for K/V, hybrid full-rank retention, recent-query-weighted reconstruction error, and low-rank attention. | Prompt-adaptive and periodically updated. It does not derive a fixed metric from model weights. |
| [STAR-KV](https://arxiv.org/abs/2606.08382) | Learned soft thresholding for rank selection, separate key/value decompositions, mixed precision, factor absorption, and fused direct kernels. | Uses calibration and optimization. It occupies adaptive low-rank factor layouts and direct execution, not the data-free metric. |
| [ReCalKV](https://arxiv.org/abs/2505.24357) | SVD of K/V projection matrices, calibration-corrected value factors, and value-factor fusion into `W_O`. | Calibration-based weight decomposition rather than runtime cache-tile shrinkage. |
| [Swift-SVD](https://arxiv.org/abs/2604.01609) | Closed-form activation-aware low-rank compression from output covariance, including KV-cache reduction. | Requires activation covariance. It reinforces that covariance-weighted SVD is occupied broadly. |
| [CARE](https://openreview.net/pdf/744fa03515a3bc1f98be734cdad93e1157b8b1f0.pdf) | Covariance-weighted SVD and curvature-aware rank allocation for converting MHA/GQA models to latent attention. | Requires calibration and targets model conversion. Its paper explicitly leaves data-free calibration as future work. |
| [LatentLLM](https://arxiv.org/abs/2505.18413) | Activation-aware and joint attention-aware tensor decomposition for converting pretrained attention to a latent representation. | Uses calibration activations and changes the model representation rather than compressing sealed cache tiles. |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) and [QuaRot](https://arxiv.org/abs/2404.00456) | Exact equivalent scaling or rotation folded through model weights to improve quantization, including KV quantization in QuaRot. | They do not supply a weighted spectral cache-block shrinker, but they occupy the change-of-basis and weight-folding algebra. |

Two additional records narrow the claim further. The 2026 Georgia Tech thesis
on [Low-rank Attention Matching](https://repository.gatech.edu/server/api/core/bitstreams/a28d359c-e0cd-41c7-bdf1-cbc4e5099896/content)
describes attention-importance-weighted SVD for keys and an attention-output
basis for values. It is not the same frozen-weight construction, but it makes a
broad "attention-weighted SVD" claim unsafe. StiefAttention minimizes full
decoder-layer output reconstruction while learning orthonormal KV bases, which
also occupies downstream-output-aware low-rank optimization even though it
requires post-training data.
[StiefAttention](https://arxiv.org/abs/2601.21686)

## What could still be materially novel

The smallest defensible research claim would combine all of these properties:

1. A data-free metric derived from frozen `W_Q`, `W_O`, normalization gains,
   GQA sharing, and an explicit future-relative-position prior.
2. A generalized spiked random-matrix model and shrinker optimized for that
   metric, not ordinary SVD performed after an arbitrary rescaling.
3. Joint rank and residual-bit allocation under a complete three-bit record
   budget, including norms, codebooks, ranks, offsets, and alignment.
4. Direct consumption of quantized factors and residuals without dense K/V
   reconstruction, with traffic lower than the current fixed K4/V2 path.
5. Evidence that the frozen metric improves unseen-model attention error,
   perplexity, retrieval, and native Apple decode speed over Euclidean
   eOptShrinkQ and fixed K4/V2 controls.

Items 1 and 2 are the scientific contribution. Items 3 through 5 are needed
to make it a LaplaceKV result rather than a metric proposal. Factor folding,
direct low-rank attention, Apple-specific SIMD, or using `W_Q` and `W_O`
individually are not sufficient novelty claims.

## Required first screen

Do not modify production model weights yet. On the existing D64 trace:

1. Build `M_K` for each KV head from every sharing query head and average its
   RoPE conjugates over a declared future-distance distribution.
2. Build `M_V` from the corresponding block of `W_O`.
3. Compare Euclidean eOptShrinkQ, simple weighted truncated SVD, and a weighted
   shrinker using identical ranks, factor precision, residual precision, and
   storage accounting.
4. Report exact attention error, K-logit error, post-`W_O` output error, factor
   traffic, and the complete record rate.
5. Reject the direction unless the weight-only metric beats Euclidean
   eOptShrinkQ and K4/V2 on the same trace without using trace-derived tuning.

This screen also tests the main risk: `W_Q^T W_Q` assumes isotropic normalized
hidden states, while OSCAR and Expected Attention both use actual activation
statistics. If the fixed proxy does not improve the trace without calibration,
there is no reason to build a native kernel.

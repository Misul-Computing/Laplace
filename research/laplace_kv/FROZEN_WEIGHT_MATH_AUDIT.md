# Frozen-weight metric math audit

Status: valid as a Qwen2.5 weight-only proxy after the qualifications below.
It is not an exact attention loss and does not by itself define a generalized
spiked-model shrinker.

## Tensor and normalization contract

Laplace decodes a GGUF matrix with dimensions `[input, output]` as contiguous
output rows. For a query head, the stored block is therefore `Wq[D,H]` and the
runtime computes, in column notation,

```text
q0 = Wq diag(g) z + b,
```

where `z` is the normalized hidden state and `g` is `attn_norm.weight`. Under
the declared zero-mean isotropic proxy `E[zz^T]=I`, the correct pre-RoPE second
moment is

```text
Cq = Wq diag(g^2) Wq^T + b b^T.
```

The input RMSNorm gain and Q projection bias must both be included. RMSNorm
has no additive bias. The extractor includes the squared attention factor
`1/D`. It is one global scalar for a fixed head dimension and would be
harmless to the basis after trace normalization. Qwen3 and Gemma-style
per-head Q RMSNorm are nonlinear in the projected activation; multiplying by
their gain is not an exact
weight-only covariance and must be labeled as another approximation. Phi-3
uses `1 + weight`, not `weight`.

The stored output projection is `Wo[H,Hq*D]`. For query head `h`, its strided
input-column block is `Oh[H,D]`, giving the local value-coordinate metric

```text
Mv,h = Oh^T Oh.
```

For KV head `j`, the key proxy must sum or average `Cq` over every contiguous
query head in its GQA group. The value proxy must likewise use every matching
`Mv,h`, not one block. Sum and average differ only by a scalar when ranks and
head weights are fixed. The value result is still a proxy: because one value
error is reused by all query heads, the true output loss contains cross-head
terms weighted by their different attention rows. Frozen weights alone do not
determine those terms.

`extract_weight_metrics.cpp` follows these Qwen2.5 orientations and grouping
rules. Its synthetic-row check agrees with the runtime dequantizer and matmul.

## RoPE averaging

Let `R_p` be the engine's NeoX-style rotation and `Cq` the pre-RoPE query
second moment. For an error represented directly in the post-RoPE coordinate
system, an absolute-query-position prior `pi(p)` gives

```text
Mabs = sum_p pi(p) R_p Cq R_p^T.
```

The extractor computes this expression for a uniform prior over metadata
positions `0..context-1`; its block formula is checked against direct dense
conjugation. This uses no trace queries or prompt statistics.

Laplace's `.kvt` capture hook records K, V, queries, and attention outputs
inside the WH-domain storage and attention path. A trace screen must therefore
use `Mwh = H Mabs H^T` and apply it directly to the captured vectors. It must
not rotate those vectors a second time. A screen on ordinary post-RoPE vectors
uses `Mabs` instead. This domain choice must be explicit in every result.

A future-relative-position prior is different. For token row `t`, it gives

```text
Mt = R_t [sum_delta pi(delta) R_delta Cq R_delta^T] R_t^T.
```

Thus one relative-position `D by D` metric cannot weight every row of a
post-RoPE tile unless the averaged metric commutes with all `R_t`. Otherwise
the loss is row-dependent. Inverse-RoPE key storage makes the relative metric
common, but ordinary direct-factor attention no longer applies because each
token needs its own rotation. Any report must call the extractor's metric an
absolute-position average, not a relative-RoPE metric.

## Weighted low rank and shrinkage

For a symmetric PSD metric `M`, the deterministic problem

```text
min rank(C)<=r ||(X-C) M^(1/2)||F
```

has the minimum-norm solution

```text
S = M^(1/2)
C = [X S]_r S^+,
```

where `[.]_r` is the truncated SVD and `S^+` is the pseudoinverse on a declared
numerical support. This remains valid for singular `M`; arbitrary components
in its nullspace are unpenalized and should be set to zero. Eigenvalue
clipping, support tolerance, and trace normalization must be fixed before
looking at trace results.

This is weighted truncated SVD, not a generalized eOptShrink rule. If the
original residual noise is isotropic, right multiplication by `S` makes it
column-correlated with covariance `M`. Applying the ordinary i.i.d. bulk edge,
D-transform, or singular-value shrinker to `X S` and mapping back silently
violates the noise model. A defensible generalized spiked-model result needs a
colored-noise law plus direction-dependent spike-overlap and weighted-risk
derivations. A changed singular-value formula fitted on the Laplace trace
would be calibration leakage, not a training-free derivation.

The safe first screen is therefore fixed-rank weighted truncated SVD against
rank-matched Euclidean truncated SVD and Euclidean eOptShrinkQ. It can reject
the weight metric, but a win cannot be described as a new generalized
shrinker.

## Fixed-basis per-token residual candidate

For a TQ2 decoded row `x0`, stored FP16 row norm `nh`, and an orthonormal row
basis `B`, the consistent correction is

```text
e = x - x0
a = (e / nh) B^T
xhat = x0 + nh Qc(a) B.
```

The stored norm is applied once. Projecting the unnormalized `e` and then
multiplying its decoded coefficient by `nh` applies the norm twice. Using the
exact encoder norm in `a` while the decoder uses `nh` also leaves an avoidable
FP16 mismatch; the zero-norm row needs the fixed all-zero path.

The ordinary TQ2 codebook is matched to coordinates with standard deviation
`1/sqrt(D)`. Under the same isotropic Gaussian model, a normalized TQ residual
has coordinate variance

```text
sigma_e^2 = E[(z - Q2(z))^2],
```

so a generic orthonormal residual coefficient has standard deviation
`sigma_e`, which is smaller. Reusing the unscaled base codebook is not the
analytic choice. A trace-free control can scale a standard four-level Lloyd
table by `sigma_e`, derived solely from the declared source model and base
quantizer. The scale must not be fitted on captured residuals.

With exact coefficients, using the leading eigenvectors of `M` is optimal
only under the additional assumption that the normalized residual covariance
is proportional to the identity. For general covariance `Sigma_e`, the
optimal fixed subspace is obtained from the leading eigenspace of
`M^(1/2) Sigma_e M^(1/2)` and mapped back through `M^(-1/2)`; coefficients are
the corresponding `M`-weighted projection. `W_Q` and `W_O` do not determine
`Sigma_e`. It would require activation calibration, which is forbidden, or a
separate frozen-`W_K`/`W_V` source-and-quantizer model declared before trace
evaluation. A basis used in the Hadamard domain must use `H M H^T`; a basis
used after inverse rotation must use `M`.

At D64, two-bit K/V bodies, two FP16 norms, and 48 two-bit coefficients do
sum to 384 bits per token, exactly 3 bits over 128 K/V scalars. That count
excludes the model-specific bases. FP16 bases with `rK+rV=48` cost
`48*64*16` bits per layer/KV head, adding `384/T` bits per cache scalar at
context `T`, or 0.09375 bits at 4096 tokens. Because the protocol counts
decoder tables, this is not an exactly-three-bit representation unless those
bases are replaced or folded into already-counted model weights without an
extra resident copy. An arbitrary post-RoPE key basis cannot be folded into
`W_Q` because the position rotation lies between the projection and basis;
only a basis commuting with every RoPE block has that escape. Its per-token
body traffic equals K4/V2 while adding basis transforms, so speed superiority
also remains unproven.

A structured alternative is to upgrade selected coordinates of the existing
Walsh-Hadamard representation from Q2 to Q4:

```text
z = x / nh
y = z H^T
yhat_j = Q4(y_j) if j in S, otherwise Q2(y_j)
xhat = nh yhat H.
```

Here `nh` is the stored FP16 norm on both encoder and decoder sides, with an
all-zero path when it is zero. The Q4 symbol carries the Q2 base information
plus two extra bit planes, so there is no separately scaled residual codebook.
These equations take `x` in the ordinary post-RoPE domain. A `.kvt` trace row
is already `y`; its simulator must quantize that row directly and must not
apply `H` again. The decoder still needs only one inverse Hadamard transform.
Under an
isotropic, equal-variance source model, the best subset of size `r` within
this restricted family is given by the largest diagonal entries of
`H M H^T`. This is not the leading eigenspace of `M` and ignores its
off-diagonal structure; it must be reported as structured mixed precision,
not weighted PCA or generalized shrinkage.

The subset remains model-, layer-, and head-specific. Two D64 masks cost 128
bits per layer/KV head and are decoder-table overhead unless recomputed for
every use. Rank sum 48 already consumes the full 3-bit body, so no mask fits.
Rank sum 47 leaves 32 raw bytes in a 128-token pair record: base codes are
4096 bytes, coefficients 1504, norms 512, and two masks 16, totaling 6128
bytes and aligning to 6144 bytes, exactly 3 bits per scalar if the fixed-size
record needs no offset or header. A universal implicit subset needs no mask
but gives up the claimed weight sensitivity. Rank allocation between K and V
must also be fixed before evaluation; their raw metric eigenvalues are in
different units and cannot be pooled without a declared normalization. This
is only a sealed-record count. An FP16 or FP32 mutable tail exceeds the gate;
the per-token candidate needs an immediately packed tail and context-by-context
accounting including its partial-byte and alignment costs.

## Leakage and evidence boundary

The metric may use frozen GGUF weights, architecture metadata, an input
isotropy assumption, and a position prior declared before evaluation. It may
not select the prior, eigenvalue threshold, rank, factor precision, residual
bits, or metric mixture from captured queries or quality results. Rank must be
held fixed for the metric ablation or selected by a separate fully specified
rule whose storage is counted.

The current `.kvt` trace format does not bind a model hash. The local GGUF has
SHA-256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`;
matching it to the existing trace is provenance by convention, not evidence
inside the artifact. Regenerate the trace from that file or extend the capture
metadata before using the result as a publication claim. The GGUF uses Q5_0
Q and output projections, so this screen measures the frozen geometry actually
used by Laplace, not the unavailable BF16 source geometry.

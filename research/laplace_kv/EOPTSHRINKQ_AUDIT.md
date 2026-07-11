# eOptShrinkQ reproduction audit

Status: rejected as a LaplaceKV candidate. The public construction can fit
below 3 bits per scalar at long D64 contexts after real metadata is counted,
but it does not beat the existing K4/V2 quality point and does not improve on
ordinary rank-matched SVD in attention error. It is also published prior art,
so it cannot supply LaplaceKV's required novelty.

## Sources and reproduction boundary

The implementation follows the algorithms and experiment settings in
[eOptShrinkQ](https://arxiv.org/abs/2605.02905), the complete D-transform
estimator in the cited
[eOptShrink paper](https://arxiv.org/abs/2207.03466), and the cited author's
[public MATLAB implementation](https://github.com/PeiChunSu/eOptShrink).
The residual quantizer follows TQ_MSE from
[TurboQuant](https://arxiv.org/abs/2504.19874): one FP16 row norm, a shared
Haar rotation, and a Gaussian Lloyd-Max scalar codebook.

An exact byte-for-byte reproduction of the eOptShrinkQ experiments is not
publicly possible as of July 12, 2026. The paper says its PyTorch code will be
released, but no official eOptShrinkQ repository was found. These details are
missing or contradictory:

- The paper defines `tilde lambda = tilde sigma^2`, then squares `tilde
  lambda` again in Algorithm 2. The prototype interprets the algorithm's
  squared terms as eigenvalues, consistent with the cited eOptShrink work.
- Algorithm 2 uses the scale-free rank test `lambda / edge - 1 > d^(-1/3)`.
  The cited paper uses an additive threshold. The scale-free eOptShrinkQ rule
  is the default here.
- The eOptShrinkQ imputation uses `(j/k)^(2/3)` for `j=1,...,k`. The cited
  eOptShrink derivation and MATLAB use the zero-based equivalent
  `((j-1)/k)^(2/3)`. Both are implemented as `--imputation paper` and
  `--imputation cited`.
- The displayed cited-paper equation for the companion Stieltjes transform
  has the wrong sign for its zero-eigenvalue mass. The author's MATLAB uses
  the minus sign required by the stated transform `int (x-z)^-1 dF(x)`.
  The prototype uses that implementation-consistent sign.
- "Adaptive Lloyd-Max" does not specify whether U and V use one codebook per
  block, one each, or one per singular vector. The measured codec uses one
  shared 16-centroid FP16 codebook per matrix. A separate U/V codebook count
  is also reported. The exact-factor result removes this ambiguity as an
  optimistic fidelity bound.
- The Haar draw, seed, precision, and portable table representation are not
  specified. The main result uses a fixed PCG64 seed and counts a full FP32
  64 by 64 decoder table. Four additional fixed seeds test sensitivity.
- The paper does not say whether the residual is formed before or after factor
  quantization. Algorithm 3 subtracts the eOptShrink estimate before the
  optional factor representation is decoded, which is the interpretation
  used here.
- No rank field, factor codebook, packing, offset, alignment, mutable-tail, or
  decoder-workspace layout is specified.

The result is therefore the strongest faithful public reproduction plus
explicit optimistic and conservative bounds. It is not presented as the
authors' unreleased implementation.

## Implemented pipeline

Each K and V block is 128 by 64. The standalone prototype performs these
steps independently for every block:

1. Compute the full SVD and its 64 observed eigenvalues.
2. Set `c=min(1/2.01, 1/log(log(64)))` and `k=floor(64^c)=7`.
3. Estimate the bulk edge and effective BBP rank with Algorithm 2.
4. Remove the outliers, impute the seven perturbed noise eigenvalues, and
   construct the empirical separable-noise distribution.
5. Compute both empirical Stieltjes transforms, their derivatives, the
   D-transform, signal strengths, overlaps, and Frobenius-optimal shrunken
   singular values.
6. Form the exact residual `R = X - U diag(phi) V^T`.
7. Quantize every residual row with two-bit TQ_MSE. Norms are rounded to FP16.
   The four Gaussian Lloyd-Max reconstruction levels and a shared Haar
   rotation are used by both encoder and decoder.
8. Quantize U and V with a data-adaptive 16-level Lloyd-Max codebook, store
   their four-bit indices, store the codebook and FP16 singular values, then
   reconstruct the low-rank and residual branches.

The ordinary-SVD control uses the same automatically selected rank, same
factor codec, same TQ residual codec, and same storage. Its only difference is
retaining the observed singular values instead of applying eOptShrink. The
exact-factor control keeps the eOptShrink factors unquantized and supplies an
optimistic bound on any four-bit factor representation.

Reproduce the main result with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  research/laplace_kv/prototype_eoptshrinkq.py \
  /tmp/laplace-qwen-256.kvt
```

## D64 quality result

The trace contains 336 queries over 48 layer/head identities and 256 tokens
per identity. Inputs and the exact reference are rounded to FP16. The first
screen keeps tokens 0 through 127 exact and compresses the completed block at
128 through 255, matching the existing sealed-tile research screen.

### Sealed tile

| Method | K score | K reconstruction | V reconstruction | Attention mean | P95 | Max |
|--------|--------:|-----------------:|-----------------:|---------------:|----:|----:|
| Exact FP16 | 0% | 0% | 0% | 0% | 0% | 0% |
| K4/V2 Q8 | 2.580% | 4.508% | 48.590% | 24.003% | 50.615% | 77.044% |
| TQ2 | 24.132% | 33.985% | 33.761% | 79.191% | 237.915% | 793.720% |
| Rank-matched SVD + TQ2 | 12.144% | 18.306% | 24.883% | 38.646% | 97.872% | 291.052% |
| eOptShrinkQ, Q4 factors | 12.104% | 18.237% | 24.729% | 39.886% | 101.285% | 302.483% |
| eOptShrinkQ, exact factors | 8.965% | 13.983% | 22.574% | 36.195% | 93.970% | 413.720% |

### Full prefix

| Method | K score | K reconstruction | V reconstruction | Attention mean | P95 | Max |
|--------|--------:|-----------------:|-----------------:|---------------:|----:|----:|
| Exact FP16 | 0% | 0% | 0% | 0% | 0% | 0% |
| K4/V2 Q8 | 4.755% | 4.809% | 49.156% | 41.865% | 87.971% | 137.450% |
| TQ2 | 32.908% | 33.925% | 33.713% | 97.610% | 262.060% | 1308.937% |
| Rank-matched SVD + TQ2 | 16.950% | 18.959% | 26.317% | 58.054% | 137.854% | 449.744% |
| eOptShrinkQ, Q4 factors | 16.875% | 18.883% | 26.157% | 57.732% | 141.761% | 490.348% |
| eOptShrinkQ, exact factors | 12.874% | 14.901% | 24.435% | 46.531% | 105.355% | 430.766% |

eOptShrink slightly improves matrix reconstruction over truncated SVD. Its
Q4-factor attention mean is 1.24 points worse on the sealed tile and 0.32
points better on the full prefix. Both are far worse than K4/V2. The
optimistic exact-factor branch is still 12.19 points worse than K4/V2 on the
sealed screen.

The paper-versus-cited imputation ambiguity does not change the decision. The
cited form measured 40.638% sealed and 58.511% full-prefix mean attention
error with Q4 factors. Choosing a transform seed after seeing this trace would
be calibration and is invalid under the training-free protocol. Neither
declared imputation beats K4/V2 on the sealed screen.

These trace errors are diagnostics, not perplexity or retrieval measurements.
They are sufficient to reject promotion because the proposed mechanism does
not improve its rank-matched SVD control and its optimistic factor bound does
not clear the existing trace reference. No production lifecycle path was
added.

## Complete storage count

The paper's displayed rate counts only two-bit residual indices and four-bit
U/V factor indices. Its later remark discloses FP16 row norms and singular
values but still does not include the adaptive codebook or a usable record
layout.

The addressable prototype record contains:

- two-bit residual codes for both K and V;
- one FP16 residual norm per token for both matrices;
- four-bit U and V factor codes for each matrix;
- one FP16 singular value per retained component;
- one 16-entry FP16 adaptive factor codebook per matrix with nonzero rank;
- a four-byte pair header containing both ranks and format flags;
- 64-byte record alignment; and
- one four-byte offset per pair record.

The ranks over 96 full-prefix tile pairs were K mean 2.948, P95 5, maximum 6,
and V mean 2.656, P95 4, maximum 5.

| Count | Mean bits/scalar | P95 | Maximum | Tiles at most 3 bits |
|-------|-----------------:|----:|--------:|----------------------:|
| Paper formula | 2.262695 | 2.375000 | 2.468750 | 100% |
| Plus disclosed norms and singulars | 2.518168 | 2.632812 | 2.728516 | 100% |
| Addressable, one factor codebook per matrix | 2.569661 | 2.689453 | 2.783203 | 100% |
| Addressable, separate U/V codebooks | 2.600911 | 2.720703 | 2.814453 | 100% |

The mean addressable record is 5,262.67 bytes. A portable D64 decoder also
needs a 16,384-byte FP32 Haar table and a 16-byte FP32 TQ codebook. Amortized
over this 96-tile trace, that shared 16,400-byte table adds 0.083415 bit per
scalar, bringing the primary mean to 2.653076. The NumPy prototype actually
uses a 32,800-byte FP64 table and reports that fact separately. Regenerating
an FP32 table from an eight-byte seed would be smaller, but no portable,
bit-exact QR construction is specified by the paper, so the storage result
does not assume it for free.

The current K4/V2 Q8 research record is 6,744 raw bytes or 6,788 bytes with
the same alignment and offset convention, equal to 3.314453 bits per scalar.

The most recent incomplete block remains FP16 as required by the paper. Its
worst D64 size is 32,512 bytes for 127 K/V rows. Including the shared table,
mean measured records, and this tail gives:

| Context | Tail | Effective bits/scalar |
|--------:|-----:|----------------------:|
| 4,095 | 127 | 3.236488 |
| 16,383 | 127 | 2.736337 |
| 65,535 | 127 | 2.611329 |

The candidate therefore meets the hard three-bit storage limit only after
enough sealed blocks amortize the decoder and mutable tail. It is not a
universal at-most-three-bit cache at short contexts.

## Direct-factor Apple attention model

Let each stored matrix be `X = U B + R`, where `B = diag(phi) V^T`. Direct
attention does not need to materialize reconstructed K or V:

- Key scores: `U (B q) + R q`.
- Value output: `(w^T U) B + w^T R`.

For the TQ residual, rotate the query once, perform all code-domain key dots,
accumulate values in the rotated domain, and inverse-rotate the final value
accumulator once. With mean measured ranks, one 128-token K/V tile pair has:

- 5,262.67 bytes of cache traffic, versus 32,768 for FP16 and 6,788 for the
  K4/V2 Q8 research record;
- 16,384 residual multiply-accumulates;
- about 1,076 low-rank multiply-accumulates; and
- 8,192 dense-rotation multiply-accumulates once per head and decode step,
  shared across every tile in that head.

At 16K and 64K, this is 6.23 times less cache traffic than FP16 and about 1.07
times its multiply-accumulate count. Relative to K4/V2 Q8, it reads 1.29 times
fewer bytes but adds dense rotations, rank-dependent branches, packed
two-bit Lloyd lookup, per-row norms, and adaptive factor lookup. Those costs
are not represented by the simple FMA count, so this model is not a speed
claim. A native kernel would still be required to prove either decode gate.

The lower-bound decode scratch is about 1,292 bytes per active head: 128
FP32 weights, rotated query/value/output vectors, and factor temporaries.
Online softmax can avoid a separate logit array, but the weights must survive
the key pass for the value pass.

## Decision

Do not promote eOptShrinkQ into LaplaceKV and do not spend a production native
kernel on this representation:

1. The fully counted long-context storage result is promising but does not
   satisfy the at-most-three-bit invariant at short context.
2. The exact paper mechanism is worse than its ordinary SVD control in the
   metric that matters here, attention output.
3. Its optimistic exact-factor bound still loses to K4/V2 on the isolated
   sealed-tile screen.
4. The measured Q4-factor format is far from evidence for the two-percent
   model-quality gate, and the public material is insufficient to reproduce
   the authors' downstream numbers exactly.
5. The mechanism is already eOptShrinkQ. Even a successful port would be an
   implementation of published work, not a novel LaplaceKV foundation.

The useful retained result is narrower: a low-rank direct-attention branch
can reduce traffic without reconstructing dense KV, but spectral shrinkage
plus TQ2 is not the residual representation to pair with it on this D64 trace.

# July 2026 public KV frontier audit

Date: 2026-07-12

## Decision

No public method found through the July 9 arXiv batch reports a point that
passes Laplace's complete three-bit storage, two-percent per-model quality,
training-free, and direct-decode gates together.

GSRQ is the new low-rate headline and MosaicKV is the strongest new systems
result. Neither is an eligible Laplace foundation. GSRQ learns calibrated
codebooks, omits those codebooks from its nominal rate, and loses substantial
quality at the accelerated low-rate point. MosaicKV reports a measured 3x
memory reduction, equivalent to 5.33 bits per scalar from FP16.

## Comparison

| Method | Reported or derived rate | Quality evidence | Decode evidence | Laplace gate failure |
|--------|--------------------------|------------------|-----------------|----------------------|
| [GSRQ](https://arxiv.org/abs/2607.01065) | 0.375 to 2 nominal bits per activation; learned codebooks excluded | Below two nominal bits, perplexity and LongBench miss the limit | Fused Triton, 1.59x to 3.40x at 8K to 64K | Calibration, learned codebooks, incomplete accounting, low-rate quality |
| [MosaicKV](https://arxiv.org/abs/2607.00760) | Measured 3x, about 5.33 bits/scalar | 1.76% average loss, up to 3.55% by tested model | Up to 4.8x lower time between tokens and 7.3x throughput | Storage and per-model quality |
| [RaBitQCache](https://arxiv.org/abs/2606.31519) | 0.5625-bit GPU selector at D128, 16.5625 total with host FP16 KV | Close on aggregate long-context tasks; GSM8K drops 4.94% | 2.16x end to end, up to 3.88x decode | Retains complete FP16 KV in host memory |
| [MixKVQ](https://arxiv.org/abs/2512.19206) | 2.3 to 2.7 nominal; metadata and exact buffers excluded | Reasoning averages drop 2.52% to 3.50% | 2.63x to 2.81x saturated-batch throughput | Model-specific search, quality, incomplete accounting |
| [TriAttention](https://arxiv.org/abs/2604.04921) | 1.50 nominal at the 10.7x headline before indices and metadata | Some matched points pass; default AIME25 drops 19.4% | Up to 2.5x at matched AIME25 accuracy | Offline calibration, pruning, no dedicated kernel |
| [KVpop](https://arxiv.org/abs/2607.05061) | 1.92 nominal bits at 88% pruning | Retains 97% performance; two-percent point uses four bits | No systems speed result | Learned future-attention scorer and quality |
| [DepthWeave-KV](https://arxiv.org/abs/2607.06523) | Claimed 8.3x, about 1.93 bits/scalar | Aggregate loss 1.41%; Needle drops 2.63% | Claims 72.8 versus 42.1 token/s | Learned routing and incomplete reproducibility |

## GSRQ accounting

GSRQ uses gain-shape K-means inside multi-stage residual vector
quantization. Its displayed rate is:

```text
BPA = R log2(K) / D
```

with `K = 256`. This counts stage indices but not the codebooks. The 0.75-BPA
configuration uses `D = 128` and `R = 12`. If each per-layer, per-subspace
codebook entry is only FP16, the codebook term alone is:

```text
R * 256 * 16 / T bits/scalar
```

The optimistic complete rate would then be:

| Context | Nominal indices | FP16 codebooks | Conditional total |
|--------:|----------------:|---------------:|------------------:|
| 8K | 0.75 | 6.00 | 6.75 |
| 32K | 0.75 | 1.50 | 2.25 |
| 64K | 0.75 | 0.75 | 1.50 |

The paper does not declare a complete codebook dtype and storage count, so
these are conditional lower bounds, not corrected paper results.

At two nominal bits, Llama 3 LongBench is 39.62 versus 40.30, but WikiText-2
perplexity rises 6.68% and C4 rises 4.65%. At 0.75 nominal bits, LongBench is
29.72 versus 40.30 for Llama 3 and 14.96 versus 43.59 for Qwen3. The fused
speed result is attached to this failing-quality point. The codebooks are
calibrated on sampled activations and use gradient information, which also
violates the training-free gate.

GSRQ is a required learned-codebook baseline. It is not evidence for a
universal sub-one-bit cache.

## MosaicKV boundary

MosaicKV combines per-vector element selection, segment-dependent SVD or
channel policies, packed sparse attention, and asynchronous CPU/GPU
recompression. Its measured footprint includes the bitmap, per-block keys,
and SVD state. The result is still only 3x FP16 compression.

Its kernel and tiering design are relevant to Laplace as systems references.
The paper says quantization can be composed with its structural reduction,
but does not establish quality after that composition. A composition cannot
inherit the three-bit or two-percent claims from either component.

## Retrieval indexes are not cache compression

RaBitQCache keeps a binary key index plus one FP16 correction factor on the
GPU while retaining the complete FP16 K/V cache in host memory. At D128, the
index rate averaged over K and V scalars is:

```text
(128 + 16) / (2 * 128) = 0.5625 bits/scalar
```

Adding the host FP16 cache gives 16.5625 bits per scalar before allocator and
transfer state. The [official
implementation](https://github.com/Sakuraaa0/RaBitQCache) also skips the first
two layers. It may be useful as an SSD or RAM retrieval index, but must never
be reported as the physical KV compression ratio.

## Other current headlines

- [Fractal KV Archives](https://arxiv.org/abs/2607.07144) is lossless only
  relative to already-lossy VQ symbols. Its 36x to 54x GPT-2 result costs 11%
  to 15% perplexity and has no attention kernel. [Official
  code](https://github.com/eighteight/fractal-kv)
- [FreqDepthKV](https://arxiv.org/abs/2607.06519) reports 3.9x, equivalent to
  about 4.10 bits per scalar, and does not close the quality evidence gap.
- [Lynx](https://arxiv.org/abs/2607.01831) progressively transfers coarse and
  residual bit streams and verifies results. It reduces exposed network
  latency, not total persistent KV storage.

## Additional mechanism closures

[VQKV](https://arxiv.org/abs/2603.16435) occupies the broad training-free
vector-quantization claim. It reports 82.8% compression and 98.6% of baseline
LongBench performance on Llama 3.1 8B. The
[official repository](https://github.com/LUMIA-Group/VQKV) lists codebook
training code and downstream evaluation code as future releases. The public
artifact therefore cannot reproduce its complete codebook lifecycle or serve
as a frozen same-machine baseline yet.

[HQMQ](https://arxiv.org/abs/2605.27646) occupies calibration-free algebraic
four-dimensional product quantization. Its near-FP16 perplexity points are
around five bits per scalar, and its downstream matched point is 3.79 bits on
Mistral. These are strong vector-codec references but remain above the
Laplace storage gate.

[Quantized Keys Steal Attention](https://arxiv.org/abs/2605.26266) identifies
the softmax Jensen bias from key quantization and corrects each attention score
from the key step sizes and query norm without extra cache storage. Its
evidence is INT2 video diffusion, not unchanged language-model KV decode, but
the paper occupies the broad zero-cache-memory Jensen-correction mechanism.

[The risk of KV cache compression](https://arxiv.org/abs/2607.01520) states
that accurate compression can be impossible in the worst case and relates the
minimax risk to intrinsic cache compressibility. Laplace can test an empirical
cross-model claim. It cannot claim a distribution-free finite-rate guarantee.

## Action

- Add GSRQ to the learned-codebook baseline set.
- Add MosaicKV to the direct packed structural baseline set.
- Add VQKV and HQMQ to the vector-codec baseline set.
- Treat analytic Jensen correction as prior art, not a standalone novelty
  claim.
- Treat RaBitQCache as an optional retrieval-index reference, not a codec.
- Do not port any current method as LaplaceKV.
- Require complete codebook, index, exact-buffer, and host-tier accounting for
  every future public rate comparison.

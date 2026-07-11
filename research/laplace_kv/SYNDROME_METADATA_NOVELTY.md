# Syndrome-carried quantization metadata novelty audit

Date: 2026-07-12

This is a technical prior-art screen, not a legal opinion or a freedom-to-
operate opinion. The narrow concept screened here is:

1. Quantize a K/V tile to fixed-width code indices.
2. Encode the tile's own inverse-quantization metadata, such as scale and zero
   point, in parities or a syndrome of those indices instead of storing a
   separate metadata field.
3. Recover the metadata from the packed indices and consume the indices in a
   direct attention kernel without materializing dense K/V.

## Decision

The broad parity or syndrome idea is not novel. Quantization index modulation,
matrix embedding, wet-paper coding, and video sign-data hiding all predate KV
caches. The closest prior art already changes quantized coefficients so that a
parity-check matrix carries omitted side information. Fixed-width quantized KV
and fused attention decode are also established separately.

No source found in this search applies a parity-check syndrome specifically to
carry a KV tile's own scale and zero-point fields. vLLM 0.25 does, however,
hide a four-bit KV zero point inside the stored float32 scale. Broadly hiding
KV quantization metadata inside another field is therefore directly occupied.
The narrower syndrome combination may be new as an application, but it is not
a safe foundational novelty claim. It looks like a direct combination of
known information embedding and known KV attention techniques. A paper would
need a new constrained quantizer or kernel co-design and a measured result that
the known combination does not predict.

## Claim chart

| Candidate element | Closest primary prior art | Assessment |
|---|---|---|
| Select among nearby quantization indices to carry information | Chen and Wornell, [Quantization Index Modulation](https://dsp-group.mit.edu/wp-content/uploads/2024/11/QuantizationIndexMod.pdf), IEEE TIT 2001 | Direct conceptual collision |
| Use a linear syndrome or parity matrix while minimizing changes to quantized coefficients | Fridrich, Goljan, and Soukal, [Perturbed Quantization with Wet Paper Codes](https://doi.org/10.1145/1022431.1022435), 2004; [EP2675159B1](https://patents.google.com/patent/EP2675159B1/en), priority 2012 | Direct mechanism collision |
| Put codec side information inside quantized coefficient bits instead of a side field | Ito and Makino, [Data hiding is a better way for transmitting side information for MP3 bitstream](https://doi.org/10.1109/IIH-MSP.2009.55), 2009 | Direct purpose collision |
| Omit syntax and derive it from parity of quantized levels | Clare and Henry, Sign Data Hiding, JCTVC-G271, 2011, summarized in [US9313498B2](https://patents.google.com/patent/US9313498B2/en) | Very close collision |
| Carry several omitted bits in overlapping parity subsets, represented as `H b = s` | [EP2675159B1](https://patents.google.com/patent/EP2675159B1/en) and [US9578347B2](https://patents.google.com/patent/US9578347B2/en) | Closest combinatorial collision |
| Let quantization-index parity determine decoder quantizer state | Haase et al., [Dependent Scalar Quantization for Neural Network Compression](https://doi.org/10.1109/ICIP40778.2020.9190955), ICIP 2020 | Neural-compression collision |
| Training-free low-bit KV with packed fixed-width indices | Liu et al., [KIVI](https://arxiv.org/abs/2402.02750), 2024; Zhang et al., [KV Cache is 1 Bit Per Channel](https://proceedings.neurips.cc/paper_files/paper/2024/file/05d6b5b6901fb57d2c287e1d3ce6d63c-Paper-Conference.pdf), NeurIPS 2024 | Direct KV format collision |
| Fuse unpacking or dequantization with attention instead of writing dense K/V | KIVI; [MiniKV](https://aclanthology.org/2025.findings-acl.952.pdf), 2025 | Direct kernel collision |
| Remove conventional scale and zero-point overhead from compressed KV | Zandieh et al., [QJL](https://arxiv.org/abs/2406.03482), 2024; TurboQuant and PolarQuant | Same objective, different mechanism |
| Store a KV zero point inside another quantization field and consume the packed cache directly | [vLLM 0.25 INT4 per-token-head KV](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/ops/int4_per_token_head/), 2026 | Direct collision for metadata-field co-packing; different from code-index syndromes |

## The strongest collision

HEVC sign-data hiding does more than watermark a signal. It removes a sign bit
from the coded representation. The encoder changes a quantized coefficient
when needed so that the parity of a group of quantized levels equals the
omitted sign. The decoder recovers the sign from that parity. This is the same
high-level bargain as removing a scale field and recovering it from code-index
parity: fewer explicit metadata bits in exchange for extra quantization
distortion.

The BlackBerry multi-bit family goes further. Its disclosure constructs binary
matrices for three- and four-bit hiding, uses partially overlapping subsets of
quantized coefficients, and describes changing a coefficient to satisfy the
required parity vector. It also says that information other than signs can be
hidden. Its granted claims are framed around encoded video and hidden sign
bits, so the text found here does not establish that a KV implementation would
infringe. The disclosure is still strong prior art against a broad claim to
syndrome-carried quantization metadata.

## Other relevant collisions

Chen and Wornell formalized quantization index modulation in 2001. A payload
selects one of several quantizer code sets, and the encoder chooses a nearby
code point in the selected set. This covers the basic act of paying distortion
to carry bits through quantization choices.

Perturbed Quantization applied wet-paper codes while a source was undergoing a
lossy quantization step. Wet positions let the encoder avoid coefficients that
must not be changed. That maps directly to saturated, high-sensitivity, or
otherwise protected K/V codes.

The 2009 MP3 work is especially relevant to the motivation. It embeds codec
side information in least-significant bits of quantized MDCT coefficients so
that the bits do not need a separate ancillary field.

Dependent scalar quantization for neural-network compression is not payload
embedding, but it makes the decoder's quantizer state a deterministic function
of preceding index parity. It weakens any broad claim that neural quantization
indices have not previously been made self-describing through parity.

KIVI, the NeurIPS channel quantization work, MiniKV, QJL, and TurboQuant occupy
the other half of the proposed combination. They establish packed low-bit KV,
training-free quantization, fused or direct attention consumption, and removal
of conventional quantization-constant overhead by other means.

vLLM 0.25 is the closest current KV-specific metadata collision. Its INT4 path
clears the low four bits of each float32 scale representation and inserts the
four-bit asymmetric zero point there. The fused paged attention kernel extracts
both fields while reading packed K/V. It does not use parity checks or change
code indices to carry a payload, so it does not anticipate the exact syndrome
channel. It does prevent any broad claim to hiding KV codec metadata inside an
existing field. The full implementation audit is in `VLLM_025_AUDIT.md`.

## Patent screen

| Family | Relevant disclosure | Initial risk reading |
|---|---|---|
| [US9313498B2](https://patents.google.com/patent/US9313498B2/en), priority 2012 | Infer an omitted sign from parity of quantized transform levels and adjust a coefficient to make parity match | Active US family, but claims are video-transform specific |
| [EP2675159B1](https://patents.google.com/patent/EP2675159B1/en), priority 2012 | Multi-bit information hiding with overlapping coefficient subsets and binary transform matrix `H` | Active EP right reported through 2032; granted claims are video and sign specific, disclosure is broader |
| [WO2005119655A1](https://patents.google.com/patent/WO2005119655A1/en) | Auxiliary data embedded by even or odd quantizer sets with distortion control | Media and perceptual-model framing; strong method prior art, less direct claim scope |
| [US12326920B2](https://patents.justia.com/patent/12326920) | Metadata embedded after training in transformed neural-network weights, optionally with channel codes | Claims concern trained weights and watermark extraction, not transient KV activations |
| [WO2021123438A1](https://patents.google.com/patent/WO2021123438A1/en) | Neural-network parameter coding in which index parity drives state transitions | Relevant to parity-driven neural quantization, not arbitrary KV metadata |

A real freedom-to-operate review must inspect claims by jurisdiction, family
continuations, prosecution history, ownership, and current legal status. This
search is not a substitute for that work.

## What may still be defensible

A defensible technical contribution would have to be narrower than "put the
metadata in parity." The strongest candidate is a complete, measured system
with all of these properties:

- the payload is specifically the same tile's inverse-quantization parameters,
  not an unrelated watermark or omitted sign;
- a cost-aware constrained quantizer minimizes attention error, handles
  saturated or protected codes, and guarantees that every legal tile has a
  solution;
- the parity-check layout matches K4/V2 bitplanes and AArch64 SIMD lanes;
- metadata recovery does not require a second full read of the tile;
- the fixed-width layout needs no variable-length directory or entropy state;
- the attention kernel consumes the packed representation directly; and
- total bytes, quality, and decode speed pass the LaplaceKV gates on the full
  model matrix.

Even that should be described as a KV-specific co-design, not as invention of
matrix embedding, syndrome coding, self-describing quantization, or direct
quantized attention.

## Technical risks before implementation

1. Same-tile global syndrome recovery normally requires reading all indices
   before the scale is known. Reading them again for attention can erase the
   bandwidth benefit. A prefix-carried or staged construction avoids the full
   second pass but concentrates index changes and is closer to old LSB side-
   information embedding.
2. K4/V2 already uses exactly three body bits per scalar. Hidden metadata can
   remove the side field, but it does not leave room for a directory, alignment,
   mutable-tail state, or fallback records unless the cache layout makes those
   costs zero or saves bits elsewhere.
3. A parity change means moving at least one code to a neighboring
   reconstruction level. With two-bit values that step is large. Wet-paper or
   weighted matrix embedding can choose cheaper carriers, but cannot make the
   distortion free.
4. A syndrome provides storage capacity, not compression. The metadata bits
   are paid for as additional quantization distortion. Quality must be tested
   through corrected autoregressive cache lifecycle runs.
5. Fixed-width decode removes rANS serialization, but speed over FP16 remains
   an empirical Apple Silicon kernel question.

## Search conclusion

No paper claim is justified from the concept alone. The exact KV-specific
syndrome combination was not located, but its ingredients and its central
parity trick are old and closely documented. vLLM now directly occupies broad
KV metadata-field co-packing. Continue only as an empirical candidate. If a
new constrained quantizer and one-pass NEON layout produce an unexpected
quality and speed result, the implementation-level combination may support a
narrow paper contribution. It does not support a broad claim of a new coding
principle or KV metadata hiding.

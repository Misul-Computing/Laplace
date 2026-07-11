# LaplaceKV bounded candidate L3-B

Status: rejected. Its K4/V2 mutable tail caused a 515.67% perplexity increase
on Qwen2.5.

L3-B keeps the sealed 128-token format, factored even-dimension transform,
Q8 metadata, two-context four-lane rANS stream, and fixed K3/V2 tile overflow
record from `CANDIDATE.md`.

The only algorithm change is the mutable tail. Each new transformed K vector
uses symmetric midrise K4 and each V vector uses symmetric midrise V2, with one
FP16 scale per vector. The tail encoder must entropy-code the six raw code bits
per K/V scalar pair enough to pay for both scales and its framing. If a token
record exceeds the exact 3-bit budget, it uses the fixed K3/V2 tail record.
The overflow count and positions are part of every result.

At 128 tokens, the sealed encoder consumes only the reconstructed K4/V2 tail.
No FP16 or FP32 copy is retained. All other gates in `CANDIDATE.md` remain
unchanged.

The FP32 cache simulator measures the quantization lifecycle only. It does not
prove that a tail entropy record fits, that overflows are rare, or that native
decode is fast.

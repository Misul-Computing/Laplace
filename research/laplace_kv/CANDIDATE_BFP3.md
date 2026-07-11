# LaplaceKV rotating BFP3 candidate

Status: rejected. Qwen2.5 perplexity increased by 13,063.82% at 512 predictions.

This candidate has no mutable tail, sealing step, entropy code, per-model
calibration, or variable record. Every transformed K and V vector is divided
into 32-coordinate blocks. Each block stores:

- one signed 4-bit power-of-two exponent
- twenty-eight symmetric midrise 3-bit codes
- four symmetric midrise 2-bit codes

The four low-bit positions are eight coordinates apart and rotate with token
position. A 16-coordinate remainder stores twelve 3-bit and four 2-bit codes.
Both layouts use exactly 3 bits per scalar, including their exponent. Token and
block positions determine the precision pattern, so there is no mask, offset,
scale table, alignment loss, or fallback mode.

The factored even-dimension transform is self-inverse. A native decoder can
unpack fixed bitplanes and apply power-of-two scaling without a serial entropy
state. This first FP32-cache simulator tests reconstructed model quality only.
Native memory and speed remain unproven.

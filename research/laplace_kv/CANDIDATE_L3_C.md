# LaplaceKV bounded candidate L3-C

Status: rejected. Its K3/V3 mutable tail caused a 450.65% perplexity increase
on Qwen2.5 at 512 predictions.

L3-C changes only the mutable-tail quantizer to symmetric midrise K3/V3 with
one FP16 scale per vector. Its raw codes already consume 3 bits per scalar, so
the tail codec must recover the scale and framing cost through a fixed,
training-free entropy code. Any over-budget token uses a lower-rate record and
is counted.

The completed tail is reconstructed and passed to the same sealed K4/V2 Q8
tile encoder. There is no hidden FP16 or FP32 copy. The cache simulator tests
this quantization lifecycle but does not prove its packed size or speed.

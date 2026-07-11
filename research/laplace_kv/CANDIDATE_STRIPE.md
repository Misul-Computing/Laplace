# LaplaceKV striped fixed-width candidate

Status: rejected in a pre-lifecycle-fix screen. Qwen2.5 perplexity increased by
19.96% at 1,024 predictions, but the exact number is not a result of the
corrected simulator.

This branch removes entropy decoding. Sealed 128-token tiles use the same
factored transform and eight-iteration variance normalization, but key rows
alternate deterministically between 3-bit and 4-bit asymmetric codes. Values
use 2-bit asymmetric codes. The fixed key pattern needs no mask or index.

The six metadata fields use linear Q6 codes with FP16 low and step headers.
At D64 the fixed code body is 2.75 bits per scalar and metadata adds 0.222656,
for 2.972656 bits per scalar before the cache-level directory. The format must
use the remaining 56 bytes per tile for its directory and alignment.

The intended native path decodes fixed bitplanes with NEON and has no serial
entropy state. This first simulator tests quality only. Mutable-tail storage,
native packed size, and speed remain unresolved, so this is not a full-cache
candidate yet.

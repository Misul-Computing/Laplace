# L3-B fixed tail prefix-code audit

Status: D64 storage passes, but the optimistic K4/V2 tail lifecycle was
rejected at +515.67% Qwen2.5 perplexity before selective fallback was added.
Native decode speed is not established.

## Format

Each transformed token stores two FP16 scales followed by one bitstream. The
first bit selects the representation:

- `0`: fixed canonical Huffman K4/V2
- `1`: fixed-width K3/V2 overflow

The decoder knows the head dimension, so both forms are self-terminating after
exactly D key symbols and D value symbols. The final byte is padded with zero
bits. There is no histogram, codebook, payload length, or per-record header.

The K4 lengths by symbol are:

```text
6 6 5 4 4 4 3 3 3 3 4 4 4 5 6 6
```

This is a complete symmetric prefix tree. The four central symbols use three
bits and the maximum length is six bits. The V2 lengths are `3 2 1 3`; the
larger central symbol wins the deterministic tie. Canonical assignment fixes
the exact bit patterns. The prototype prints them and performs an exact decode
roundtrip for every record.

The table is fixed in the format and the encoder does not fit a distribution
at runtime. It was selected on this D64 trace before cross-model testing, so it
must be treated as frozen and trace-informed rather than universal evidence.

## D64 measurement

Both `/tmp/laplace-qwen-256.kvt` and
`/tmp/laplace-qwen-256-all.kvt` contain the same 12,288 K/V token records. The
results are identical.

| Quantity | Result |
|---|---:|
| Hard record cap | 48 bytes |
| Entropy candidate mean | 43.656820 bytes |
| Entropy candidate p95 | 46 bytes |
| Entropy candidate maximum | 51 bytes |
| Stored mean after K3/V2 overflow | 43.643880 bytes |
| Stored p95 | 46 bytes |
| Stored maximum | 48 bytes |
| Stored mean rate | 2.727743 bits/scalar |
| K3/V2 overflows | 38 / 12,288 |
| Overflow rate | 0.309245% |

The K3/V2 fallback is 45 bytes at D64, including both FP16 scales, the mode
bit, and byte padding. Every stored record therefore fits the 48-byte cap.

The fallback positions are not uniformly distributed. Twenty-eight of the 38
occur in layer 6, head 0. A low global fallback percentage does not by itself
show low model impact. The selective fallback lifecycle needs an end-to-end
quality run.

## Dimension formulas

For dimension D, the exact 3-bit-per-scalar record cap is:

```text
cap(D) = 6D / 8 bytes
```

An entropy record passes when its K and V prefix codes use at most:

```text
entropy_code_bits <= 6D - 33
```

The 33 bits are two FP16 scales and one mode bit. The fixed fallback size,
including byte padding, is:

```text
fallback(D) = ceil((5D + 33) / 8) bytes
```

| Dimension | Cap | Entropy-code limit | K3/V2 fallback | Byte margin |
|---:|---:|---:|---:|---:|
| 64 | 48 | 351 bits | 45 | 3 |
| 96 | 72 | 543 bits | 65 | 7 |
| 256 | 192 | 1503 bits | 165 | 27 |

These are budget formulas, not D96 or D256 measurements. The fallback proves a
record cap for D64, D96, and D256, but not a quality bound. D32 would need 25
bytes against a 24-byte cap and therefore needs another format.

Packed records are sequentially self-delimiting. This audit does not count a
random-access offset array. Fixed 48-byte D64 slots need no offsets but consume
exactly 3 bits per scalar and discard the measured average saving. Any offset
index retained by the cache must be included in whole-cache accounting.

Reproduce with:

```sh
python3 research/laplace_kv/prototype_tail_huffman.py \
  /tmp/laplace-qwen-256.kvt /tmp/laplace-qwen-256-all.kvt
```

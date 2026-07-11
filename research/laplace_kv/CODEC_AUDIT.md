# LaplaceKV tile codec audit

This audit tests byte-exact compression of the released dense KVarN K4/V2
codes and FP16 metadata. It also tests the Q8 metadata candidate separately.
The input traces are `/tmp/laplace-qwen-256.kvt` and
`/tmp/laplace-qwen-256-all.kvt`. Both traces yield the same 48 unique sealed
D64/G128 tiles after duplicate query views are removed.

## Exact FP16 metadata result

Each tile contains 8,192 K4 codes, 8,192 V2 codes, and 576 FP16 metadata
values. The raw FP16 metadata is 1,152 bytes. The tested codecs preserve the
official codes and every metadata bit.

| Codec | Mean bits/scalar | p95 | Max |
|---|---:|---:|---:|
| Context rANS codes, LZMA zigzag metadata | 2.9585 | 3.1379 | 3.1641 |
| Context rANS codes, zlib zigzag metadata | 2.9604 | 3.1508 | 3.1719 |
| Context rANS codes, native LZFSE zigzag metadata | 2.9985 | 3.1820 | 3.1953 |

The best average result is below 3 bits per scalar, but no tested lossless
FP16-metadata format keeps every tile below 3. LZMA is also a poor fit for a
decode-time Apple CPU kernel. Native LZFSE and LZ4 do not close the tail gap.
Simple horizontal and vertical delta, XOR, ULEB128, byte-plane, and nibble
rANS variants also fail the hard limit.

## Four-state context stream with Q8 metadata

The useful code correlation is along the token axis. The compact stream uses
four interleaved rANS states for K and four for V. Codes are stored token
major, so attention can consume decoded codes without constructing a dense
tile.

K selects one of two probability models using the top bit of the prior
token's same-dimension K4 code. V selects one of two models using the top bit
of the prior token's same-dimension V2 code. The model tables contain the
first 15 K frequencies and first 3 V frequencies as uint16 values. The last
frequency follows from the fixed total.

The fully counted tile contains:

- rANS payloads, including eight 32-bit initial states
- 72 bytes of normalized frequency tables
- 16 bytes of internal absolute stream offsets and format fields
- one 8-byte absolute tile-directory offset
- 600 bytes of Q8 per-field metadata, including six FP16 low/step pairs
- padding to a 16-byte tile boundary

All encode and decode roundtrips are checked by the prototype. The observed
rate is 2.859375 mean, 2.973828 p95, and 2.992188 maximum bits per scalar.
All 48 captured tiles are below 3.

Q8 metadata is lossy relative to official FP16 metadata. This result proves a
storage format on one captured D64/G128 tile set. It does not prove model
quality or cross-model universality. The later native fused benchmark rejected
the CPU rANS path at 35x to 40x slower than FP16. Entropy coding by itself is
not a novelty claim.

Reproduce the measurements with:

```sh
python3 research/laplace_kv/prototype_lossless_metadata.py TRACE.kvt
```

#!/usr/bin/env python3
"""Read GGUF metadata and tensor names from a (possibly partial) GGUF file."""
import struct
import sys

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# GGUF v2/v3 metadata value types (matches the engine's GGUFValueType).
SCALAR_FORMATS = {
    0: '<B',   # UINT8
    1: '<b',   # INT8
    2: '<H',   # UINT16
    3: '<h',   # INT16
    4: '<I',   # UINT32
    5: '<i',   # INT32
    6: '<f',   # FLOAT32
    7: '<B',   # BOOL
    8: None,   # STRING
    9: None,   # ARRAY
    10: '<Q',  # UINT64
    11: '<q',  # INT64
    12: '<d',  # FLOAT64
}

# Tensor types the engine decodes (see GGMLType in src/tensor.h).
TENSOR_TYPE_NAMES = {
    0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
    8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K',
    14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 19: 'IQ1_S', 30: 'BF16',
    31: 'I8', 32: 'I32', 33: 'I64', 34: 'U8', 35: 'U32', 36: 'BOOL',
    100: 'GROUPED_AFFINE_U2_256', 101: 'COLUMN_GROUPED_AFFINE_U2_SKIP_256',
}

# Sanity caps so a corrupt header cannot make this tool read forever.
MAX_STRING_BYTES = 1 << 30
MAX_ARRAY_ELEMS = 10_000_000


class GGUFError(Exception):
    pass


def read_exact(f, n, what):
    buf = f.read(n)
    if len(buf) != n:
        raise GGUFError(f"truncated file: wanted {n} bytes for {what}, got {len(buf)}")
    return buf


def read_str(f):
    n = struct.unpack('<Q', read_exact(f, 8, 'string length'))[0]
    if n > MAX_STRING_BYTES:
        raise GGUFError(f"string length {n} exceeds the inspection limit")
    return read_exact(f, n, 'string bytes').decode('utf-8', errors='replace')


def read_value(f, vtype):
    if vtype == 8:
        return read_str(f)
    if vtype == 9:
        atype = struct.unpack('<I', read_exact(f, 4, 'array type'))[0]
        n = struct.unpack('<Q', read_exact(f, 8, 'array length'))[0]
        if n > MAX_ARRAY_ELEMS:
            raise GGUFError(f"array length {n} exceeds the inspection limit")
        return [read_value(f, atype) for _ in range(n)]
    fmt = SCALAR_FORMATS.get(vtype)
    if fmt is None:
        raise GGUFError(f"unknown metadata type {vtype}")
    return struct.unpack(fmt, read_exact(f, struct.calcsize(fmt), 'metadata value'))[0]


def inspect(path, f):
    magic = struct.unpack('<I', read_exact(f, 4, 'magic'))[0]
    if magic != GGUF_MAGIC:
        raise GGUFError(f"not a GGUF file (bad magic {magic:#x})")
    version = struct.unpack('<I', read_exact(f, 4, 'version'))[0]
    if not 2 <= version <= 3:
        raise GGUFError(f"unsupported GGUF version {version} (need 2 or 3)")
    n_tensors = struct.unpack('<Q', read_exact(f, 8, 'tensor count'))[0]
    n_kv = struct.unpack('<Q', read_exact(f, 8, 'metadata count'))[0]
    print(f"GGUF v{version}, {n_tensors} tensors, {n_kv} metadata keys")
    print("\n=== METADATA ===")
    for _ in range(n_kv):
        key = read_str(f)
        vtype = struct.unpack('<I', read_exact(f, 4, 'metadata type'))[0]
        print(f"  {key} = {read_value(f, vtype)}")
    print(f"\n=== TENSORS ({n_tensors}) ===")
    for i in range(n_tensors):
        try:
            name = read_str(f)
            n_dims = struct.unpack('<I', read_exact(f, 4, 'dimension count'))[0]
            dims = [struct.unpack('<Q', read_exact(f, 8, 'dimension'))[0]
                    for _ in range(n_dims)]
            ttype = struct.unpack('<I', read_exact(f, 4, 'tensor type'))[0]
            offset = struct.unpack('<Q', read_exact(f, 8, 'tensor offset'))[0]
        except GGUFError as error:
            # The metadata section is what matters for partial downloads; the
            # tensor list simply ends wherever the file was cut off.
            print(f"  (tensor list truncated after {i}/{n_tensors}: {error})",
                  file=sys.stderr)
            return
        tname = TENSOR_TYPE_NAMES.get(ttype, f'type{ttype}')
        print(f"  {name}: {tname} dims={dims} offset={offset}")


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} <model.gguf>", file=sys.stderr)
        return 2
    path = argv[1]
    try:
        with open(path, 'rb') as f:
            inspect(path, f)
    except OSError as error:
        print(f"error: cannot read {path}: {error.strerror or error}", file=sys.stderr)
        return 1
    except GGUFError as error:
        print(f"error: {path}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

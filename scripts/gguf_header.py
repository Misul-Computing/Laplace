#!/usr/bin/env python3
"""Read GGUF metadata and tensor names from a (possibly partial) GGUF file."""
import struct, sys

def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8')

def main(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == 0x46554747, f"bad magic {magic:#x}"
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        print(f"GGUF v{version}, {n_tensors} tensors, {n_kv} metadata keys")
        print("\n=== METADATA ===")
        for _ in range(n_kv):
            key = read_str(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            if vtype == 0:   val = struct.unpack('<B', f.read(1))[0]
            elif vtype == 1: val = struct.unpack('<b', f.read(1))[0]
            elif vtype == 2: val = struct.unpack('<H', f.read(2))[0]
            elif vtype == 3: val = struct.unpack('<h', f.read(2))[0]
            elif vtype == 4: val = struct.unpack('<I', f.read(4))[0]
            elif vtype == 5: val = struct.unpack('<i', f.read(4))[0]
            elif vtype == 6: val = struct.unpack('<f', f.read(4))[0]
            elif vtype == 7: val = struct.unpack('<B', f.read(1))[0]  # BOOL
            elif vtype == 8: val = read_str(f)
            elif vtype == 9:
                atype = struct.unpack('<I', f.read(4))[0]
                n = struct.unpack('<Q', f.read(8))[0]
                if atype in (4,5): val = list(struct.unpack(f'<{n}I', f.read(4*n)))
                elif atype in (2,3): val = list(struct.unpack(f'<{n}H', f.read(2*n)))
                elif atype in (10,11): val = list(struct.unpack(f'<{n}Q', f.read(8*n)))
                elif atype == 8: val = [read_str(f) for _ in range(n)]
                elif atype == 7: val = list(struct.unpack(f'<{n}B', f.read(n)))  # BOOL array
                elif atype == 6: val = list(struct.unpack(f'<{n}f', f.read(4*n)))
                else: val = f"(array type {atype}, {n} elems)"
            else:
                val = f"(unknown type {vtype})"
            print(f"  {key} = {val}")
        print(f"\n=== TENSORS ({n_tensors}) ===")
        type_names = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',
                      10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',30:'BF16'}
        for _ in range(n_tensors):
            name = read_str(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tname = type_names.get(ttype, f'type{ttype}')
            print(f"  {name}: {tname} dims={dims} offset={offset}")

if __name__ == '__main__':
    main(sys.argv[1])

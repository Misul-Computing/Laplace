#!/usr/bin/env python3

import hashlib
import os
from pathlib import Path
import random
import secrets
import struct
import subprocess
import sys
import time


NO_VALUE = 0xFFFFFFFF
NO_PLANE = 0xFFFF
NO_POLICY = 0xFFFF


def section(tag: int, body: bytes) -> bytes:
    return struct.pack("<HI", tag, len(body)) + body


def half_to_float_bits(value: int) -> int:
    sign = (value & 0x8000) << 16
    exponent = (value >> 10) & 0x1F
    fraction = value & 0x03FF
    if exponent == 0:
        if fraction == 0:
            return sign
        shift = 0
        while (fraction & 0x0400) == 0:
            fraction <<= 1
            shift += 1
        fraction &= 0x03FF
        return sign | ((113 - shift) << 23) | (fraction << 13)
    if exponent == 0x1F:
        return sign | 0x7F800000 | (fraction << 13)
    return sign | ((exponent + 112) << 23) | (fraction << 13)


def instruction(opcode: int, value_type: int, operands=(), *, immediate=0,
                plane=NO_PLANE, policy=NO_POLICY, width=0, bit_order=1) -> bytes:
    values = list(operands) + [NO_VALUE] * (3 - len(operands))
    return struct.pack(
        "<HBBIIIHHQB7x", opcode, value_type, bit_order, *values,
        plane, policy, immediate, width)


def encode_program(rank: int, result: int, planes_count: int,
                   policies: bytes, nodes: list[bytes]) -> bytes:
    metadata = struct.pack("<HBBI", 1, rank, 0, result)
    planes = struct.pack("<I", planes_count)
    for _ in range(planes_count):
        planes += struct.pack("<BBHIQQ", 1, 0, 0, 1, 0, 0)
    instructions = struct.pack("<I", len(nodes)) + b"".join(nodes)
    body = b"".join((
        section(1, metadata), section(2, planes), section(3, policies),
        section(4, instructions), section(5, b"")))
    return b"LAPPHY01" + struct.pack("<HHI", 1, 5, 16 + len(body)) + body


def codebook_case(rng: random.Random) -> tuple[bytes, list[bytes], int, int, tuple[int, ...], tuple[int, ...]]:
    # The test binary is already linked when this program is created. The
    # values, tail, and table contents are deliberately generated here rather
    # than copied from a C++ fixture.
    extents = (rng.randint(2, 5), rng.randint(5, 11))
    coordinate = (rng.randrange(extents[0]), rng.randrange(extents[1]))
    width = rng.randint(2, 4)
    entries = 1 << width
    divisor = rng.randint(2, 5)
    while divisor == width:
        divisor = rng.randint(2, 5)
    codes = [rng.randrange(entries)
             for index in range(extents[0] * extents[1])]
    selected = rng.randrange(entries)
    codes[coordinate[1] * extents[0] + coordinate[0]] = selected
    packed = bytearray((len(codes) * width + 7) // 8)
    for index, code in enumerate(codes):
        address = index * width
        for bit in range(width):
            packed[(address + bit) // 8] |= ((code >> bit) & 1) << ((address + bit) % 8)

    half_pool = (0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0xC100,
                 0xC400, 0x7BFF, 0x3800, 0xBC00, 0x4500, 0xC600,
                 0x3400, 0x3A00, 0x4100, 0xC300)
    half_table = tuple(rng.sample(half_pool, entries))
    table = struct.pack(f"<{entries}H", *half_table)

    # Typed SSA: axis permutation + exact quotient/remainder tail + dynamic
    # codebook address. No source-format tag is present in the wire.
    nodes = [
        instruction(5, 2, immediate=1),          # coordinate axis 1
        instruction(10, 2, (0,), immediate=divisor),
        instruction(1, 2, immediate=divisor),
        instruction(9, 2, (1, 2)),
        instruction(11, 2, (0,), immediate=divisor),
        instruction(7, 2, (3, 4)),               # recomposed coordinate
        instruction(6, 2, immediate=0),          # extent axis 0
        instruction(9, 2, (5, 6)),
        instruction(5, 2, immediate=0),          # coordinate axis 0
        instruction(7, 2, (7, 8)),
        instruction(1, 2, immediate=width),
        instruction(9, 2, (9, 10)),              # packed bit address
        instruction(16, 3, (11,), plane=0, width=width),
        instruction(13, 2, (12,)),
        instruction(1, 2, immediate=16),
        instruction(9, 2, (13, 14)),             # table bit address
        instruction(16, 3, (15,), plane=1, width=16),
        instruction(23, 5, (16,), policy=0),
    ]
    policies = struct.pack("<I8B", 1, 1, 1, 1, 1, 1, 1, 1, 0)
    wire = encode_program(2, 17, 2, policies, nodes)
    expected = half_to_float_bits(half_table[selected])
    return wire, [bytes(packed), table], expected, 6, extents, coordinate


def signed_case(rng: random.Random) -> tuple[bytes, list[bytes], int, int, tuple[int, ...], tuple[int, ...]]:
    extent = rng.randint(7, 29)
    coordinate = rng.randrange(extent)
    width = rng.randint(3, 9)
    offset = rng.randint(1, 7)
    while offset == width:
        offset = rng.randint(1, 7)
    address = coordinate * width + offset
    packed = bytearray((extent * width + offset + 7) // 8)
    raw = (1 << (width - 1)) | rng.randrange(1 << (width - 1))
    for bit in range(width):
        packed[(address + bit) // 8] |= ((raw >> bit) & 1) << ((address + bit) % 8)
    nodes = [
        instruction(5, 2, immediate=0),
        instruction(1, 2, immediate=width),
        instruction(9, 2, (0, 1)),
        instruction(1, 2, immediate=offset),
        instruction(7, 2, (2, 3)),
        instruction(16, 3, (4,), plane=0, width=width),
        instruction(22, 4, (5,), width=width),
    ]
    wire = encode_program(1, 6, 1, struct.pack("<I", 0), nodes)
    expected = raw | (~((1 << width) - 1) & 0xFFFFFFFF)
    return wire, [bytes(packed)], expected, 2, (extent,), (coordinate,)


def msb_float_case(rng: random.Random) -> tuple[bytes, list[bytes], int, int, tuple[int, ...], tuple[int, ...]]:
    count = rng.randint(4, 12)
    coordinate = rng.randrange(count)
    values = tuple(struct.unpack("<I", struct.pack("<f", rng.uniform(-8, 8)))[0]
                   for _ in range(count))
    nodes = [
        instruction(5, 2, immediate=0),
        instruction(1, 2, immediate=32),
        instruction(9, 2, (0, 1)),
        instruction(16, 3, (2,), plane=0, width=32, bit_order=2),
        instruction(25, 5, (3,), policy=0),
    ]
    policies = struct.pack("<I8B", 1, 1, 1, 1, 1, 1, 1, 1, 0)
    wire = encode_program(1, 4, 1, policies, nodes)
    return (wire, [struct.pack(f">{count}I", *values)], values[coordinate],
            6, (count,), (coordinate,))


def fma_case(rng: random.Random) -> tuple[bytes, list[bytes], int, int, tuple[int, ...], tuple[int, ...]]:
    unit = rng.randint(1, 3)
    values = (1.0 + unit * 2.0**-23,
              1.0 + unit * 2.0**-23,
              -(1.0 + unit * 2.0**-22))
    raw = tuple(struct.unpack("<I", struct.pack("<f", value))[0]
                for value in values)
    nodes = []
    converted = []
    for index in range(3):
        address = len(nodes)
        nodes.append(instruction(1, 2, immediate=index * 32))
        loaded = len(nodes)
        nodes.append(instruction(16, 3, (address,), plane=0, width=32))
        converted.append(len(nodes))
        nodes.append(instruction(25, 5, (loaded,), policy=0))
    nodes.append(instruction(33, 5, tuple(converted), policy=1))
    policies = struct.pack(
        "<I8B8B", 2,
        1, 1, 1, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 1, 2, 0)
    wire = encode_program(0, len(nodes) - 1, 1, policies, nodes)
    expected = struct.unpack("<I", struct.pack("<f", unit * unit * 2.0**-46))[0]
    return wire, [struct.pack("<3I", *raw)], expected, 6, (), ()


def fixture_bundle(case) -> tuple[bytes, bytes]:
    wire, planes, expected, element, extents, coordinate = case
    bundle = bytearray(b"LAPFX001")
    bundle += struct.pack("<HBB", 2, element, len(extents))
    bundle += struct.pack(f"<{len(extents)}Q", *extents)
    bundle += struct.pack(f"<{len(coordinate)}Q", *coordinate)
    bundle += struct.pack("<I", len(wire))
    bundle += wire
    bundle += struct.pack("<I", len(planes))
    for plane, payload in enumerate(planes):
        bundle += struct.pack("<IQ", plane, len(payload))
        bundle += payload
    return bytes(bundle), wire


def fixture_value(stdout: str) -> int:
    for line in stdout.splitlines():
        if line.startswith("physical_value=0x"):
            return int(line.split("=", 1)[1], 16)
    raise RuntimeError("fixture binary did not report a physical value")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_unseen_physical_program.py BINARY OUTPUT_DIR")
    binary = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_mtime = binary.stat().st_mtime_ns
    executable = binary.read_bytes()
    seed_text = os.environ.get("LAPLACE_UNSEEN_SEED")
    seed = int(seed_text, 0) if seed_text else secrets.randbits(64)
    rng = random.Random(seed)
    cases = (codebook_case(rng), signed_case(rng), msb_float_case(rng),
             fma_case(rng))
    evidence = [f"seed={seed}"]
    for index, case in enumerate(cases):
        bundle, wire = fixture_bundle(case)
        fixture = output_dir / f"physical-program-unseen-{index}.fixture"
        fixture.write_bytes(bundle)
        if fixture.stat().st_mtime_ns <= binary_mtime:
            time.sleep(0.002)
            fixture.write_bytes(bundle)
        if fixture.stat().st_mtime_ns <= binary_mtime:
            raise SystemExit("fixture was not created after the test binary")
        digest = hashlib.sha256(
            b"laplace-physical-program-v1\0" +
            struct.pack("<Q", len(wire)) + wire).digest()
        if (wire in executable or digest in executable or
                digest.hex().encode() in executable):
            raise SystemExit("fixture wire or digest is embedded in the test binary")
        completed = subprocess.run(
            [str(binary), "--fixture", str(fixture)], check=False,
            capture_output=True, text=True)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        actual = fixture_value(completed.stdout)
        expected = case[2]
        if actual != expected:
            raise SystemExit(
                f"fixture {index} mismatch: expected=0x{expected:016x} "
                f"actual=0x{actual:016x}")
        evidence.append(
            f"case={index} wire_sha256={hashlib.sha256(wire).hexdigest()} "
            f"program_digest={digest.hex()} expected=0x{expected:016x}")
    evidence_path = output_dir / "physical-program-unseen-evidence.txt"
    evidence_path.write_text("\n".join(evidence) + "\n")
    print("\n".join(evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

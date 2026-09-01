#!/usr/bin/env python3
import hashlib
import pathlib
import struct
import subprocess
import sys
import tempfile


NO_REGISTER = 0xFFFFFFFF


def instruction(opcode, destination=NO_REGISTER, input_a=NO_REGISTER,
                input_b=NO_REGISTER, section=0, immediate=0, literal=b""):
    return {
        "opcode": opcode,
        "destination": destination,
        "input_a": input_a,
        "input_b": input_b,
        "section": section,
        "immediate": immediate,
        "literal": literal,
    }


def program_digest(program):
    digest = hashlib.sha256()
    digest.update(b"laplace-container-schema-program-v1")
    digest.update(struct.pack(
        "<HHIIQQII", program["major"], program["minor"],
        program["register_count"], program["predicate_count"],
        program["maximum_steps"], program["maximum_loop_iterations"],
        program["maximum_ranges"], len(program["instructions"])))
    for item in program["instructions"]:
        digest.update(struct.pack(
            "<BIIIIQI", item["opcode"], item["destination"],
            item["input_a"], item["input_b"], item["section"],
            item["immediate"], len(item["literal"])))
        digest.update(item["literal"])
    return digest.digest()


def encode_program(program):
    payload = bytearray(struct.pack(
        "<HHIIQQII", program["major"], program["minor"],
        program["register_count"], program["predicate_count"],
        program["maximum_steps"], program["maximum_loop_iterations"],
        program["maximum_ranges"], len(program["instructions"])))
    for item in program["instructions"]:
        payload.extend(struct.pack(
            "<BBBBIIIIQI", item["opcode"], 0, 0, 0,
            item["destination"], item["input_a"], item["input_b"],
            item["section"], item["immediate"], len(item["literal"])))
        payload.extend(item["literal"])
    payload.extend(program_digest(program))
    return bytes(payload)


def encode_schema_set(programs):
    records = sorted((program_digest(program), encode_program(program))
                     for program in programs)
    body = bytearray()
    for _, record in records:
        body.extend(struct.pack("<I", len(record)))
        body.extend(record)
    set_digest = hashlib.sha256()
    set_digest.update(b"laplace-container-schema-set-v1")
    set_digest.update(struct.pack("<I", len(records)))
    for digest, _ in records:
        set_digest.update(digest)
    total = 24 + len(body) + 32
    header = struct.pack("<8sHHIII", b"LAPCSW01", 1, 0, total,
                         len(records), 0)
    return header + body + set_digest.digest()


def unseen_schema():
    return {
        "major": 1,
        "minor": 0,
        "register_count": 3,
        "predicate_count": 1,
        "maximum_steps": 1_000_000,
        "maximum_loop_iterations": 1_000_000,
        "maximum_ranges": 65_536,
        "instructions": [
            instruction(1, literal=b"UXS1"),
            instruction(3, destination=0),
            instruction(2, destination=1),
            instruction(11, input_a=0),
            instruction(5, destination=2),
            instruction(12, input_a=2, input_b=1, section=23),
            instruction(10, input_a=1),
            instruction(15),
        ],
    }


def run(command, expect_success=True):
    completed = subprocess.run(command, text=True, capture_output=True)
    if expect_success != (completed.returncode == 0):
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(
            f"unexpected exit {completed.returncode}: {' '.join(command)}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_unseen_container_schema.py TEST_BINARY")
    binary = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="laplace-unseen-schema-") as root:
        root = pathlib.Path(root)
        package_path = root / "package.bin"
        schema_path = root / "schema.bin"
        container_path = root / "container.bin"
        run([binary, "--emit-program-package", str(package_path)])
        package = package_path.read_bytes()
        padding = b"post-v1"
        offset = 4 + 8 + 4 + len(padding)
        container = (b"UXS1" + struct.pack("<QI", offset, len(package)) +
                     padding + package)
        schema_path.write_bytes(encode_schema_set([unseen_schema()]))
        container_path.write_bytes(container)
        run([binary, "--load-container-schema", str(schema_path),
             str(container_path)])

        prefix_schema = unseen_schema()
        prefix_schema["instructions"].pop()
        schema_path.write_bytes(encode_schema_set([prefix_schema]))
        container_path.write_bytes(container + b"ATTACKER_TRAILER")
        run([binary, "--load-container-schema", str(schema_path),
             str(container_path)], expect_success=False)

        corrupt = bytearray(container)
        corrupt[offset] ^= 1
        container_path.write_bytes(corrupt)
        run([binary, "--load-container-schema", str(schema_path),
             str(container_path)], expect_success=False)

        bad_schema = bytearray(schema_path.read_bytes())
        bad_schema[-1] ^= 1
        schema_path.write_bytes(bad_schema)
        container_path.write_bytes(container)
        run([binary, "--load-container-schema", str(schema_path),
             str(container_path)], expect_success=False)

    print("unseen container schema: OK")


if __name__ == "__main__":
    main()

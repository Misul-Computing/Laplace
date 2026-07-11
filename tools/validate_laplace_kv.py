#!/usr/bin/env python3
"""Run a reproducible LaplaceKV perplexity and storage screen."""

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def named_path(value):
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return name, resolved


def screening_gates(result, preliminary):
    finite = all(
        not isinstance(value, float) or math.isfinite(value)
        for value in result.values()
    )
    gates = {
        "finite": finite,
        "predictions": result.get("predictions", 0) >= 2048,
        "quality": result.get("ppl_delta_pct", math.inf) <= 2.0,
        "storage": result.get("storage_valid") is True and result.get(
            "laplace_effective_bits_per_scalar", math.inf) <= 3.0,
        "not_preliminary": not preliminary and not result.get(
            "preliminary", True),
    }
    return gates, all(gates.values())


def write_record(output, record):
    output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    output.flush()
    os.fsync(output.fileno())


def self_test():
    passing = {
        "predictions": 2048,
        "ppl_delta_pct": 2.0,
        "laplace_effective_bits_per_scalar": 3.0,
        "preliminary": False,
    }
    passing["storage_valid"] = True
    gates, passed = screening_gates(passing, False)
    assert passed and all(gates.values())
    failing = dict(passing, laplace_effective_bits_per_scalar=3.0001)
    gates, passed = screening_gates(failing, False)
    assert not passed and not gates["storage"]
    gates, passed = screening_gates(dict(passing, storage_valid=False), False)
    assert not passed and not gates["storage"]
    print("self-test passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--laplace", default="build/laplace")
    parser.add_argument("--model", action="append", type=named_path, default=[])
    parser.add_argument("--corpus", action="append", type=named_path, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predictions", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--baseline")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--preliminary", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.model or not args.corpus or args.output is None:
        parser.error("--model, --corpus, and --output are required")
    if args.predictions < 65:
        parser.error("--predictions must be at least 65")

    laplace = Path(args.laplace).resolve()
    if not laplace.is_file():
        parser.error(f"not a file: {laplace}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    header = {
        "type": "laplace_kv_screen_run",
        "schema": 2,
        "scope": "paired_perplexity_and_logical_storage_only",
        "full_publication_gate": False,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "laplace": str(laplace),
        "laplace_sha256": sha256(laplace),
        "predictions": args.predictions,
        "threads": args.threads,
        "baseline": args.baseline,
        "streaming": args.stream,
        "preliminary": args.preliminary,
    }
    all_screened = True
    with args.output.open("x", encoding="utf-8") as output:
        write_record(output, header)
        model_hashes = {str(path): sha256(path) for _, path in args.model}
        corpus_hashes = {str(path): sha256(path) for _, path in args.corpus}
        for model_name, model in args.model:
            for corpus_name, corpus in args.corpus:
                command = [
                    str(laplace), str(model), "--eval-file", str(corpus),
                    "--eval-limit", str(args.predictions),
                    "--max-seq", str(args.predictions),
                    "--laplace-stream" if args.stream else "--laplace-resident",
                ]
                if args.threads:
                    command.extend(["-j", str(args.threads)])
                if args.preliminary:
                    command.append("--eval-preliminary")
                print(f"running {model_name}/{corpus_name}", file=sys.stderr,
                      flush=True)
                environment = os.environ.copy()
                if args.baseline:
                    environment["LAPLACE_KV_BASELINE"] = args.baseline
                completed = subprocess.run(
                    command, text=True, capture_output=True, check=False,
                    env=environment)
                if completed.returncode:
                    write_record(output, {
                        "type": "screen_result",
                        "model_id": model_name,
                        "model_sha256": model_hashes[str(model)],
                        "corpus_id": corpus_name,
                        "corpus_sha256": corpus_hashes[str(corpus)],
                        "command": command,
                        "returncode": completed.returncode,
                        "stderr": completed.stderr,
                        "screen_passed": False,
                    })
                    all_screened = False
                    continue
                lines = [line for line in completed.stdout.splitlines()
                         if line.startswith("{")]
                if len(lines) != 1:
                    raise RuntimeError(
                        f"expected one JSON result for {model_name}/{corpus_name}")
                result = json.loads(lines[0])
                if (args.baseline
                        and result.get("research_baseline") != args.baseline):
                    raise RuntimeError(
                        f"requested baseline {args.baseline!r}, got "
                        f"{result.get('research_baseline')!r}")
                gates, passed = screening_gates(result, args.preliminary)
                result.update({
                    "type": "screen_result",
                    "model_id": model_name,
                    "model_sha256": model_hashes[str(model)],
                    "corpus_id": corpus_name,
                    "corpus_sha256": corpus_hashes[str(corpus)],
                    "command": command,
                    "stderr": completed.stderr,
                    "gates": gates,
                    "screen_passed": passed,
                })
                write_record(output, result)
                all_screened &= passed
        write_record(output, {
            "type": "screen_summary",
            "screen_passed": all_screened,
            "full_publication_gate": False,
            "finished_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
        })
    return 0 if all_screened else 1


if __name__ == "__main__":
    raise SystemExit(main())

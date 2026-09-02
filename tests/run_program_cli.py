#!/usr/bin/env python3
import pathlib
import subprocess
import sys
import tempfile


def run(command):
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(
            f"unexpected exit {completed.returncode}: {' '.join(command)}")
    return completed


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_program_cli.py LAPLACE TEST_BINARY")
    laplace, test_binary = sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="laplace-program-cli-") as root:
        root = pathlib.Path(root)
        container = root / "container.bin"
        manifest = root / "manifest.bin"
        container.touch()
        manifest.touch()
        run([test_binary, "--emit-program-ingress", str(container),
             str(manifest)])
        completed = run([
            laplace, str(container), "--program-manifest", str(manifest),
            "-p", "a", "-n", "1", "--greedy", "--max-seq", "4",
            "--raw-prompt",
        ])
        if "[program] Metal session:" not in completed.stderr:
            sys.stderr.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            raise SystemExit("program CLI did not report the verified Metal route")

    print("program CLI: OK")


if __name__ == "__main__":
    main()

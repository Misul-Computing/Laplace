#!/usr/bin/env python3
"""Download a starter GGUF model for Laplace.

Default target: Qwen2.5-0.5B-Instruct Q4_K_M (several hundred MB),
stored on the Hugging Face CDN. Use --url and --out to
fetch a different package.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

DEFAULT_URL = ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/"
               "resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf")
CHUNK_BYTES = 1 << 20  # 1 MB


def fail(out: Path, done: int, reason: str) -> None:
    if out.is_file():
        out.unlink()
    print(f"error: download failed after {done / 1e6:.1f} MB: {reason}",
          file=sys.stderr, flush=True)
    print("error: removed the partial file; re-run to retry",
          file=sys.stderr, flush=True)
    sys.exit(1)


def download(url: str, out: Path) -> None:
    if out.exists():
        print(f"error: {out} already exists; remove it or pass --out",
              file=sys.stderr)
        sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {out}", flush=True)
    print(f"  from {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "laplace-downloader/1.0"})
    done = 0
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, out.open("wb") as f:
            try:
                total = int(resp.headers.get("Content-Length", 0))
            except ValueError:
                total = 0
            while True:
                buf = resp.read(CHUNK_BYTES)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r  {done / 1e6:7.1f} / {total / 1e6:7.1f} MB"
                          f"  ({pct:5.1f}%)", end="", flush=True)
            print(flush=True)
            if total and done < total:
                fail(out, done, f"connection dropped at {done} of {total} bytes")
    except OSError as error:  # URLError, HTTPError, timeouts, resets
        fail(out, done, str(error))
    with out.open('rb') as f:
        if f.read(4) != b'GGUF':
            fail(out, done, "downloaded content is not a GGUF package "
                            "(check the URL; gated or moved models return HTML)")
    print(f"done: {out}  ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a starter GGUF model for Laplace.")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help="package URL (default: Qwen2.5-0.5B-Instruct Q4_K_M)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path (default: models/<url basename>)")
    args = parser.parse_args()
    out = args.out if args.out else Path("models") / args.url.rsplit("/", 1)[-1]
    try:
        download(args.url, out)
    except KeyboardInterrupt:
        print(f"\ninterrupted; partial file left at {out}", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()

"""Download a small LLaMA-arch GGUF for testing laplace.
Falls back to direct HTTP if huggingface_hub CAS fails."""
import sys
import urllib.request
from pathlib import Path

models_dir = Path("models").resolve()
models_dir.mkdir(parents=True, exist_ok=True)

# Qwen2.5-0.5B-Instruct: small (~400MB Q4_K_M), pure LLaMA arch with GQA + RoPE.
# HF stores these on CDN with redirect.
url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
fname = models_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

print(f"Downloading to {fname}", flush=True)
print(f"  from {url}", flush=True)

req = urllib.request.Request(url, headers={"User-Agent": "laplace-downloader/1.0"})
with urllib.request.urlopen(req, timeout=60) as resp, open(fname, "wb") as f:
    total = int(resp.headers.get("Content-Length", 0))
    chunk = 1 << 20  # 1 MB
    done = 0
    while True:
        buf = resp.read(chunk)
        if not buf:
            break
        f.write(buf)
        done += len(buf)
        if total:
            pct = 100.0 * done / total
            print(f"\r  {done/1e6:7.1f} / {total/1e6:7.1f} MB  ({pct:5.1f}%)", end="", flush=True)
    print(flush=True)

print(f"done: {fname}  ({fname.stat().st_size/1e6:.1f} MB)", flush=True)

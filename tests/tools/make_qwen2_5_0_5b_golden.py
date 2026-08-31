#!/usr/bin/env python3
"""Write the fixed Task P Qwen2.5-0.5B GGUF golden record."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


INPUT_IDS = (1, 2, 3, 4, 5, 6, 7, 8)
VOCABULARY = 151936
ATOL = 1e-4
RTOL = 1e-4


def sha256(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def parse_input_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(part) for part in value.split(","))
    if ids != INPUT_IDS:
        raise ValueError(f"input IDs must be exactly {','.join(map(str, INPUT_IDS))}")
    return ids


def write_golden(model_dir: Path, gguf: Path, output: Path, input_ids: tuple[int, ...]) -> None:
    if not model_dir.is_dir():
        raise ValueError(f"missing model directory: {model_dir}")
    if not gguf.is_file():
        raise ValueError(f"missing GGUF: {gguf}")
    with (model_dir / "config.json").open(encoding="utf-8") as config_file:
        source_config = json.load(config_file)
    if source_config.get("tie_word_embeddings") is not True:
        raise ValueError("pinned source configuration must tie input and output embeddings")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        gguf_file=str(gguf),
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.lm_head.weight = model.model.embed_tokens.weight
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise ValueError("failed to restore tied output embedding alias")
    model.eval()

    body = bytearray(
        struct.pack(
            "<7sHH32sIIff8I",
            b"LAPGLD1",
            1,
            0,
            sha256(gguf),
            len(input_ids),
            VOCABULARY,
            ATOL,
            RTOL,
            *input_ids,
        )
    )
    with torch.no_grad():
        for length in range(1, len(input_ids) + 1):
            token_ids = torch.tensor([input_ids[:length]], dtype=torch.long)
            logits = model(input_ids=token_ids, use_cache=False).logits[0, -1]
            if logits.dtype != torch.float32 or logits.numel() != VOCABULARY:
                raise ValueError("unexpected authoritative logit type or vocabulary")
            values = logits.detach().cpu().tolist()
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"non-finite logit at prefix {length}")
            body.extend(struct.pack("<II", length, int(torch.argmax(logits).item())))
            body.extend(struct.pack(f"<{VOCABULARY}f", *values))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(body + hashlib.sha256(body).digest())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-ids", required=True)
    args = parser.parse_args()
    write_golden(args.model_dir, args.gguf, args.output, parse_input_ids(args.input_ids))


if __name__ == "__main__":
    main()

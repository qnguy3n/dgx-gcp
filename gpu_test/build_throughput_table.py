"""Builds a markdown file with the two tables backing decode_throughput.png
and concurrent_throughput.png -- same data, tabular form.

Usage:
    python3 build_throughput_table.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

DATASETS = [
    ("vLLM (AWQ, DGX)", "vllm_results.csv"),
    ("Ollama (Q4_K_M, DGX)", "ollama_results.csv"),
    ("Ollama tuned (Q4_K_M, DGX)", "ollama_tuned_results.csv"),
    ("Ollama (Q4_K_M, T4)", "ollama_t4_results.csv"),
]

MODEL_ORDER = ["qwen2.5-1.5b", "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b"]
PARAMS_B = {"qwen2.5-1.5b": 1.5, "qwen2.5-7b": 7, "qwen2.5-14b": 14, "qwen2.5-32b": 32, "qwen2.5-72b": 72}


def load(filename: str) -> dict:
    path = RESULTS_DIR / filename
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["model"]: r for r in csv.DictReader(f) if r.get("status") == "ok"}


T4_NAME = "Ollama (Q4_K_M, T4)"


def fmt(v) -> str:
    if v in (None, ""):
        return "—"
    return f"{float(v):.1f}"


def build_table(metric_key: str, loaded: dict) -> list[str]:
    header = "| Model | Params (B) | " + " | ".join(name for name, _ in DATASETS) + " |"
    sep = "|---|---|" + "---|" * len(DATASETS)
    lines = [header, sep]
    for model in MODEL_ORDER:
        row = [model, str(PARAMS_B[model])]
        t4_row = loaded[T4_NAME].get(model)
        t4_val = float(t4_row[metric_key]) if t4_row else None
        for name, _ in DATASETS:
            r = loaded[name].get(model)
            if not r:
                row.append("—")
                continue
            val = float(r[metric_key])
            # Ratio vs. Ollama/T4 baseline, shown for the two DGX columns only.
            if name != T4_NAME and t4_val:
                row.append(f"{val:.1f} ({val / t4_val:.2f}x)")
            else:
                row.append(fmt(val))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_vram_table(loaded: dict) -> list[str]:
    header = "| Model | Params (B) | " + " | ".join(name for name, _ in DATASETS) + " |"
    sep = "|---|---|" + "---|" * len(DATASETS)
    lines = [header, sep]
    for model in MODEL_ORDER:
        row = [model, str(PARAMS_B[model])]
        for name, _ in DATASETS:
            r = loaded[name].get(model)
            row.append(fmt(r["model_mem_gib"]) if r else "—")
        lines.append("| " + " | ".join(row) + " |")
    return lines


# Usable VRAM per T4: 16 GB nameplate reports as ~15 GiB in nvidia-smi, minus
# ~0.5-1 GiB of CUDA context/driver/runtime overhead.
T4_USABLE_GIB = 14.0
T4_HEADROOM = 1.25  # recommended count also covers peak activation memory during the burst


def build_t4_count_table(loaded: dict) -> list[str]:
    tuned = loaded.get("Ollama tuned (Q4_K_M, DGX)", {})
    header = "| Model | VRAM needed @ conc=8 (GiB) | T4s, bare minimum | T4s, recommended (+25% headroom) |"
    sep = "|---|---|---|---|"
    lines = [header, sep]
    for model in MODEL_ORDER:
        r = tuned.get(model)
        if not r:
            lines.append(f"| {model} | — | — | — |")
            continue
        need = float(r["model_mem_gib"])
        bare = math.ceil(need / T4_USABLE_GIB)
        recommended = math.ceil(need * T4_HEADROOM / T4_USABLE_GIB)
        lines.append(f"| {model} | {need:.1f} | {bare} | {recommended} |")
    return lines


def main() -> None:
    loaded = {name: load(filename) for name, filename in DATASETS}

    lines = []
    lines.append("# GPU throughput benchmark results")
    lines.append("")
    lines.append(
        "Qwen2.5 family, 4-bit quantized, 5 sizes (1.5B-72B). "
        "Source: results/vllm_results.csv, results/ollama_results.csv, "
        "results/ollama_tuned_results.csv, results/ollama_t4_results.csv. "
        "Plots: results/plots/decode_throughput.png, results/plots/concurrent_throughput.png."
    )
    lines.append("")
    ratio_note = "Values in parentheses are the ratio vs. the Ollama (Q4_K_M, T4) column (e.g. 2.00x = twice the T4 throughput)."

    lines.append("## GPU VRAM required (GiB)")
    lines.append("")
    lines.extend(build_vram_table(loaded))
    lines.append("")
    lines.append(
        "Resident model memory as reported by each backend after load "
        "(weights + runtime overhead), captured right after a single 1-token "
        "warm-up request -- **before** the concurrency=8 benchmark runs, so "
        "these numbers are not a live measurement under concurrent load."
    )
    lines.append("")
    lines.append(
        "**Why concurrency=8 doesn't cost ~8x VRAM:** weights are loaded once "
        "and shared by every concurrent request in a batch -- they aren't "
        "duplicated per request. Only the KV cache (cached attention "
        "keys/values) scales with concurrency, and it's small relative to "
        "weights, especially with grouped-query attention (few KV heads vs. "
        "many query heads). This box's `ollama.service` sets "
        "`OLLAMA_NUM_PARALLEL=8` and `OLLAMA_CONTEXT_LENGTH=4096`, so Ollama "
        "pre-reserves KV-cache space for 8 parallel slots x 4096 tokens at "
        "load time regardless of live traffic. For qwen2.5-32b (64 layers, "
        "8 KV heads, head_dim 128, fp16 cache): "
        "`2 x 64 x 8 x 128 x 2 bytes = 256 KiB/token`, and "
        "`256 KiB x 4096 ctx x 8 slots ~= 8 GiB` -- which lines up almost "
        "exactly with the 20.0 -> 28.0 GiB jump between the Ollama (DGX) and "
        "Ollama tuned (DGX) columns above. Caveat: the same math predicts a "
        "comparable ~1.75 GiB bump for qwen2.5-7b, which wasn't observed, so "
        "this is a suggestive match rather than a fully controlled result -- "
        "the two runs were captured a day apart, not as an A/B, so measurement "
        "noise or per-model automatic sizing by Ollama can't be ruled out."
    )
    lines.append("")
    lines.append("## T4 GPUs needed for a VRAM-fair concurrency=8 comparison")
    lines.append("")
    lines.append(
        "The T4 column above ran on a single 16 GB T4 and couldn't even load "
        "qwen2.5-32b/72b, so it isn't a fair comparison at concurrency=8 -- "
        "here's how many T4s it would take to just hold what the tuned DGX "
        "run needed (weights + the `OLLAMA_NUM_PARALLEL=8`/`OLLAMA_CONTEXT_LENGTH=4096` "
        "KV-cache reservation)."
    )
    lines.append("")
    lines.extend(build_t4_count_table(loaded))
    lines.append("")
    lines.append(
        f"A T4's 16 GB nameplate reports as ~15 GiB in `nvidia-smi`, and CUDA "
        f"context/driver overhead takes another slice, so this assumes "
        f"**{T4_USABLE_GIB:.0f} GiB usable per card**. \"Bare minimum\" is "
        f"`ceil(VRAM needed / {T4_USABLE_GIB:.0f})` -- it lands models like "
        f"qwen2.5-32b exactly on a card boundary with zero slack for the "
        f"transient activation memory a concurrency=8 burst allocates on top "
        f"of the resident footprint, so it'll likely OOM in practice. "
        f"\"Recommended\" adds {int((T4_HEADROOM - 1) * 100)}% headroom for that. "
        f"Also: T4s have no NVLink, so pooling VRAM across cards for a model "
        f"that doesn't fit on one relies on Ollama's own multi-GPU layer-splitting "
        f"(pipeline-style, not tensor-parallel) -- it recovers the *capacity*, "
        f"but cross-card PCIe hops per token mean decode latency still won't "
        f"match the single-package DGX Spark's unified memory, even with "
        f"\"enough\" T4s."
    )
    lines.append("")
    lines.append("## Is a 4xT4 cloud box (e.g. GCP n1-standard-32) a fair DGX Spark comparison?")
    lines.append("")
    lines.append(
        "Checked against a concrete example: a GCP `n1-standard-32` "
        "(32 vCPUs, 120 GB host RAM) with 4xT4 attached -- 4 is also the max "
        "T4 count Google allows on a single VM (1, 2, or 4 only, never more). "
        "**VRAM fit:** 4 cards x 14 GiB usable = ~56 GiB covers 1.5b/7b/14b/32b "
        "with headroom to spare, but falls *short* of qwen2.5-72b's 57 GiB "
        "need -- i.e. the largest single-VM T4 config still can't fairly hold "
        "the biggest model in this sweep, let alone with the recommended "
        "6-card margin."
    )
    lines.append("")
    lines.append(
        "Even where VRAM fits, three confounds ride along with swapping in "
        "T4s, so this isn't a controlled, single-variable comparison:"
    )
    lines.append("")
    lines.append(
        "1. **GPU generation.** T4 is Turing (2018): 65 TFLOPS FP16, 320 GB/s "
        "GDDR6, 75 W, no native FP8/FP4 tensor engine. DGX Spark's GB10 is "
        "Blackwell (2025): ~11-12 TFLOPS FP16 measured but up to 1 PFLOP FP4 "
        "sparse, 273 GB/s unified memory. Opposite design points -- 4x the "
        "older card's dense FP16 doesn't translate to 4x the throughput that "
        "matters for a Q4_K_M-quantized workload."
    )
    lines.append(
        "2. **No NVLink.** T4s only talk over PCIe Gen3, so splitting 32b/72b "
        "across cards adds a cross-card hop per layer that DGX Spark's single "
        "coherent-memory package never pays -- decode latency won't be "
        "comparable even at matched VRAM."
    )
    lines.append(
        "3. **Memory model.** The 120 GB host RAM on the VM is separate DRAM, "
        "not usable as extra VRAM -- each T4's 16 GB is a hard wall. DGX "
        "Spark's 128 GB is one coherent pool CPU and GPU both draw from."
    )
    lines.append("")
    lines.append(
        "**Verdict:** legitimate as a \"what would I actually rent instead\" "
        "check, and it confirms 1.5b-32b run at all on cheaper/older hardware -- "
        "but it isn't a fair, single-variable hardware comparison, and it "
        "can't even host 72b with headroom. A cleaner test would size a "
        "*single* current-gen GPU with >=80 GB (e.g. an A100/H100 80GB or "
        "L40S) to hold 72b alone, isolating \"GPU generation + unified vs. "
        "discrete memory\" as the one thing that changes."
    )
    lines.append("")
    lines.append("## Single-request decode throughput (tok/s)")
    lines.append("")
    lines.extend(build_table("decode_tok_s_mean", loaded))
    lines.append("")
    lines.append(ratio_note)
    lines.append("")
    lines.append("## Concurrent throughput, concurrency=8 (tok/s)")
    lines.append("")
    lines.extend(build_table("concurrent_throughput_tok_s", loaded))
    lines.append("")
    lines.append(ratio_note)
    lines.append("")

    out_path = RESULTS_DIR / "sweep_report.md"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

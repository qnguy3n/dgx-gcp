"""Runs the standard GPU benchmark battery across the model sweep, on Ollama.

Uses Ollama's OpenAI-compatible endpoint (http://localhost:11434/v1) so
bench_client.py is identical code to the vLLM sweep -- no separate metric
logic per backend. Requires Ollama installed and its server running
(`ollama serve`, or the systemd service) before this is run.

Ollama and vLLM can't hold two large models on this GPU simultaneously
either, but that's moot here since only one process (Ollama's) ever loads
a model on this side; nothing else needs to be stopped.

Usage:
    python3 run_ollama_sweep.py                        # all 5 models
    python3 run_ollama_sweep.py --models qwen2.5-1.5b,qwen2.5-7b
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from bench_client import run_full_benchmark
from gpu_monitor import GpuMonitor
from models import MODELS, MODELS_BY_LABEL

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
LOG_DIR = RESULTS_DIR / "logs"
CSV_PATH = RESULTS_DIR / "ollama_results.csv"

BASE_URL = "http://localhost:11434"
PULL_TIMEOUT_S = 3600  # 72B GGUF is tens of GB

CSV_FIELDS = [
    "timestamp", "backend", "model", "params_b", "quantization",
    "load_time_s", "model_mem_gib",
    "ttft_ms_mean", "ttft_ms_p95", "itl_ms_mean", "decode_tok_s_mean",
    "prefill_tok_s_mean", "prompt_tokens",
    "concurrency", "concurrent_throughput_tok_s", "concurrent_mean_latency_ms",
    "gpu_util_pct_mean", "gpu_util_pct_max", "power_w_mean", "power_w_max",
    "status", "error",
]


def check_server_up() -> bool:
    try:
        return requests.get(f"{BASE_URL}/api/tags", timeout=5).ok
    except requests.RequestException:
        return False


def pull_model(tag: str, log_path: Path) -> tuple[bool, str]:
    print(f"[ollama] pulling {tag} ...")
    with open(log_path, "w") as f:
        proc = subprocess.run(["ollama", "pull", tag], stdout=f, stderr=subprocess.STDOUT,
                               timeout=PULL_TIMEOUT_S, text=True)
    if proc.returncode != 0:
        return False, f"pull failed, see {log_path}"
    return True, ""


def unload_model(tag: str) -> None:
    subprocess.run(["ollama", "stop", tag], capture_output=True, timeout=30)


def get_model_mem_gib(tag: str) -> float:
    """Parses `ollama ps` SIZE column for the currently-loaded model."""
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    for line in out.splitlines()[1:]:
        if line.startswith(tag) or tag in line.split()[0:1]:
            m = re.search(r"([\d.]+)\s*GB", line)
            if m:
                return float(m.group(1))
    return None


def warm_up_and_time_load(tag: str) -> tuple[bool, float]:
    """A trivial request forces the model to load into memory; timed as
    the load_time_s metric (mirrors what the vLLM sweep measures)."""
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={"model": tag, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            timeout=1800,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[ollama] warm-up failed for {tag}: {exc}")
        return False, time.monotonic() - t0
    return True, time.monotonic() - t0


def append_row(row: dict, csv_path: Path) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k) for k in CSV_FIELDS})


def run_one(model, concurrency: int = 8, log_prefix: str = "ollama") -> dict:
    log_path = LOG_DIR / f"{log_prefix}_{model.label}.log"
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "backend": "ollama",
        "model": model.label, "params_b": model.params_b, "quantization": "q4_k_m",
    }

    ok, err = pull_model(model.ollama_tag, log_path)
    if not ok:
        row["status"] = "pull_failed"
        row["error"] = err
        print(f"[ollama] {model.label}: {err}")
        return row

    ok, load_time_s = warm_up_and_time_load(model.ollama_tag)
    row["load_time_s"] = round(load_time_s, 1)
    row["model_mem_gib"] = get_model_mem_gib(model.ollama_tag)

    if not ok:
        row["status"] = "failed_to_start"
        row["error"] = f"see {log_path}"
        return row

    print(f"[ollama] {model.label}: up after {load_time_s:.0f}s, benchmarking...")
    monitor = GpuMonitor()
    monitor.start()
    try:
        metrics = run_full_benchmark(BASE_URL, model.ollama_tag, concurrency=concurrency)
        row.update(metrics)
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - record and move on to the next model
        row["status"] = "benchmark_error"
        row["error"] = str(exc)
        print(f"[ollama] {model.label}: benchmark error: {exc}")
    finally:
        row.update(monitor.stop())
        unload_model(model.ollama_tag)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="comma-separated model labels to run (default: all)")
    parser.add_argument("--output", default="ollama_results.csv",
                         help="CSV filename under results/ (default: ollama_results.csv)")
    parser.add_argument("--concurrency", type=int, default=8,
                         help="concurrent requests for the throughput test (default: 8)")
    args = parser.parse_args()
    selected = MODELS
    if args.models:
        wanted = args.models.split(",")
        selected = [MODELS_BY_LABEL[w] for w in wanted]
    csv_path = RESULTS_DIR / args.output

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not check_server_up():
        raise SystemExit(
            "Ollama server not reachable at http://localhost:11434 -- "
            "install it and make sure `ollama serve` (or the systemd service) is running first."
        )

    log_prefix = Path(args.output).stem  # e.g. ollama_tuned_results -> distinct log files
    for model in selected:
        row = run_one(model, concurrency=args.concurrency, log_prefix=log_prefix)
        append_row(row, csv_path)
        print(f"[ollama] {model.label}: done, status={row.get('status')}\n")


if __name__ == "__main__":
    main()

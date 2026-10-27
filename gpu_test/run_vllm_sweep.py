"""Runs the standard GPU benchmark battery across the model sweep, on vLLM.

Stops the production vLLM server (../setup_vllm_openai.txt) for the
duration -- this machine's single GPU can't hold two large models at once
(confirmed empirically: a second instance fails with a CUDA OOM even at
init, while the production container holds its memory reservation) -- and
restores it when done, including on failure/Ctrl-C.

Usage:
    python3 run_vllm_sweep.py                        # all 5 models
    python3 run_vllm_sweep.py --models qwen2.5-1.5b,qwen2.5-7b
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from bench_client import run_full_benchmark
from gpu_monitor import GpuMonitor
from models import MODELS, MODELS_BY_LABEL

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
LOG_DIR = RESULTS_DIR / "logs"
CSV_PATH = RESULTS_DIR / "vllm_results.csv"

CONTAINER = "gpu-test-vllm"
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"
MAX_MODEL_LEN = 4096
GPU_MEM_UTIL = 0.85
STARTUP_TIMEOUT_S = 1800  # 30 min -- 72B can be slow to download + load
SETUP_SCRIPT = HERE.parent / "setup_vllm_openai.txt"

CSV_FIELDS = [
    "timestamp", "backend", "model", "params_b", "quantization",
    "load_time_s", "model_mem_gib",
    "ttft_ms_mean", "ttft_ms_p95", "itl_ms_mean", "decode_tok_s_mean",
    "prefill_tok_s_mean", "prompt_tokens",
    "concurrency", "concurrent_throughput_tok_s", "concurrent_mean_latency_ms",
    "gpu_util_pct_mean", "gpu_util_pct_max", "power_w_mean", "power_w_max",
    "status", "error",
]


def sh(cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Runs a docker command with the docker group applied (this session's
    user isn't in the docker group at the OS-credential level; `sg` grants
    it for the child process without needing a fresh login)."""
    return subprocess.run(["sg", "docker", "-c", cmd], capture_output=True, text=True, timeout=timeout)


def docker_rm(name: str) -> None:
    sh(f"docker rm -f {name}")


def stop_production_server() -> bool:
    res = sh(f"docker ps --filter name=gpt-oss-vllm --format {{{{.Names}}}}")
    running = "gpt-oss-vllm" in res.stdout
    if running:
        print("[vllm] stopping production server for the duration of the sweep...")
        sh("docker stop gpt-oss-vllm", timeout=60)
        time.sleep(2)
    return running


def restore_production_server(was_running: bool) -> None:
    if not was_running:
        return
    print("[vllm] restoring production server...")
    sh("docker rm -f gpt-oss-vllm")
    subprocess.run(["sg", "docker", "-c", f"bash {SETUP_SCRIPT}"], check=False)


def start_vllm(repo: str) -> None:
    cmd = (
        f"docker run --rm -d --name {CONTAINER} --gpus all -p {PORT}:8000 "
        f"-v ~/.cache/huggingface:/root/.cache/huggingface "
        f"-v /tmp/encodings:/etc/encodings:ro "
        f"-e TIKTOKEN_ENCODINGS_BASE=/etc/encodings "
        f"vllm/vllm-openai:latest --model {repo} --quantization awq "
        f"--gpu-memory-utilization {GPU_MEM_UTIL} --max-model-len {MAX_MODEL_LEN} --host 0.0.0.0"
    )
    sh(cmd, timeout=30)


def wait_ready(log_path: Path, timeout_s: int) -> tuple[bool, float, float]:
    """Polls `docker logs` until startup completes or fails.
    Returns (ok, load_time_s, model_mem_gib)."""
    start = time.monotonic()
    mem_gib = None
    while time.monotonic() - start < timeout_s:
        res = sh(f"docker logs {CONTAINER}", timeout=15)
        logs = res.stdout + res.stderr
        log_path.write_text(logs)
        m = re.search(r"Model loading took ([\d.]+) GiB memory", logs)
        if m:
            mem_gib = float(m.group(1))
        if "Application startup complete" in logs:
            return True, time.monotonic() - start, mem_gib
        if re.search(r"Traceback|CUDA error|torch\.AcceleratorError", logs):
            return False, time.monotonic() - start, mem_gib
        time.sleep(5)
    return False, timeout_s, mem_gib


def append_row(row: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    new_file = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k) for k in CSV_FIELDS})


def run_one(model) -> dict:
    log_path = LOG_DIR / f"vllm_{model.label}.log"
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "backend": "vllm",
        "model": model.label, "params_b": model.params_b, "quantization": "awq",
    }
    docker_rm(CONTAINER)
    print(f"[vllm] starting {model.label} ({model.vllm_repo}) ...")
    start_vllm(model.vllm_repo)
    ok, load_time_s, mem_gib = wait_ready(log_path, STARTUP_TIMEOUT_S)
    row["load_time_s"] = round(load_time_s, 1)
    row["model_mem_gib"] = mem_gib

    if not ok:
        row["status"] = "failed_to_start"
        row["error"] = f"see {log_path}"
        print(f"[vllm] {model.label}: FAILED to start (see {log_path})")
        docker_rm(CONTAINER)
        return row

    print(f"[vllm] {model.label}: up after {load_time_s:.0f}s, benchmarking...")
    monitor = GpuMonitor()
    monitor.start()
    try:
        metrics = run_full_benchmark(BASE_URL, model.vllm_repo)
        row.update(metrics)
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - record and move on to the next model
        row["status"] = "benchmark_error"
        row["error"] = str(exc)
        print(f"[vllm] {model.label}: benchmark error: {exc}")
    finally:
        row.update(monitor.stop())
        docker_rm(CONTAINER)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="comma-separated model labels to run (default: all)")
    args = parser.parse_args()
    selected = MODELS
    if args.models:
        wanted = args.models.split(",")
        selected = [MODELS_BY_LABEL[w] for w in wanted]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    was_running = stop_production_server()
    try:
        for model in selected:
            row = run_one(model)
            append_row(row)
            print(f"[vllm] {model.label}: done, status={row.get('status')}\n")
    finally:
        docker_rm(CONTAINER)
        restore_production_server(was_running)


if __name__ == "__main__":
    main()

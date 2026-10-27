"""Plots model size vs. each benchmark metric, one PNG per metric, vLLM and
Ollama overlaid for direct comparison. Reads whatever of
results/{vllm,ollama}_results.csv exists -- safe to run after a partial
sweep. Only 'ok' rows are plotted; failed rows are listed on stderr.

Usage:
    python3 plot_results.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

METRICS = [
    ("ttft_ms_mean", "Time to first token (ms)", False),
    ("itl_ms_mean", "Inter-token latency (ms/token)", False),
    ("decode_tok_s_mean", "Decode throughput, single request (tok/s)", False),
    ("prefill_tok_s_mean", "Prefill throughput (tok/s)", False),
    ("concurrent_throughput_tok_s", "Aggregate throughput, concurrency=8 (tok/s)", False),
    ("concurrent_mean_latency_ms", "Mean per-request latency under load (ms)", False),
    ("load_time_s", "Model load time (s)", False),
    ("model_mem_gib", "GPU memory footprint (GiB)", False),
    ("gpu_util_pct_mean", "Mean GPU utilization (%)", False),
    ("power_w_mean", "Mean GPU power draw (W)", False),
]

BACKEND_STYLE = {
    "vllm": dict(color="#4C72B0", marker="o", label="vLLM (AWQ)"),
    "ollama": dict(color="#DD8452", marker="s", label="Ollama (Q4_K_M)"),
}


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = {
        "vllm": load_csv(RESULTS_DIR / "vllm_results.csv"),
        "ollama": load_csv(RESULTS_DIR / "ollama_results.csv"),
    }

    for backend, backend_rows in rows.items():
        failed = [r for r in backend_rows if r.get("status") != "ok"]
        for r in failed:
            print(f"[plot] skipping {backend}/{r.get('model')}: status={r.get('status')} error={r.get('error')}")

    any_plotted = False
    for metric_key, ylabel, log_y in METRICS:
        fig, ax = plt.subplots(figsize=(7, 5))
        plotted_this_metric = False

        for backend, backend_rows in rows.items():
            ok_rows = [r for r in backend_rows if r.get("status") == "ok"]
            points = [(to_float(r["params_b"]), to_float(r.get(metric_key))) for r in ok_rows]
            points = [(x, y) for x, y in points if x is not None and y is not None]
            if not points:
                continue
            points.sort(key=lambda p: p[0])
            xs, ys = zip(*points)
            style = BACKEND_STYLE[backend]
            ax.plot(xs, ys, marker=style["marker"], color=style["color"], label=style["label"], linewidth=2)
            plotted_this_metric = True
            any_plotted = True

        if not plotted_this_metric:
            plt.close(fig)
            continue

        ax.set_xscale("log")
        ax.set_xlabel("Model size (billion params)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. model size")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path = PLOTS_DIR / f"{metric_key}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {out_path}")

    if not any_plotted:
        print("[plot] no 'ok' rows found in either CSV yet -- run a sweep first.")


if __name__ == "__main__":
    main()

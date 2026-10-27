"""Focused plot: model size vs. single-request decode throughput and vs.
concurrent throughput, one PNG each, all backends overlaid. Reads every
results/*_results.csv file present (skips empty ones), so dropping in e.g.
results/ollama_t4_results.csv from another machine picks it up automatically.

X-axis uses a log scale (model sizes span ~48x) but tick labels are the
plain model-size numbers (1.5, 7, 14, 32, 72), not scientific notation.
Each point is annotated with its model label.

Usage:
    python3 plot_throughput.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

METRICS = [
    ("decode_tok_s_mean", "Single-request decode throughput (tok/s)", "decode_throughput.png"),
    ("concurrent_throughput_tok_s", "Concurrent throughput, concurrency=8 (tok/s)", "concurrent_throughput.png"),
]

# Color/marker per backend; add more here if a new CSV introduces a new backend name.
BACKEND_STYLE = {
    "vllm": dict(color="#4C72B0", marker="o", label="vLLM (AWQ, DGX)"),
    "ollama": dict(color="#DD8452", marker="s", label="Ollama (Q4_K_M, DGX)"),
    "ollama_t4": dict(color="#55A868", marker="^", label="Ollama (Q4_K_M, T4)"),
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


def backend_key_from_filename(path: Path) -> str:
    # results/vllm_results.csv -> "vllm", results/ollama_t4_results.csv -> "ollama_t4"
    return path.stem.replace("_results", "")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(RESULTS_DIR.glob("*_results.csv"))
    datasets = {}
    for path in csv_paths:
        rows = load_csv(path)
        if not rows:
            print(f"[plot] {path.name} is empty, skipping")
            continue
        datasets[backend_key_from_filename(path)] = rows

    if not datasets:
        print("[plot] no data found in any *_results.csv -- run a sweep first.")
        return

    all_x_values = set()

    for metric_key, ylabel, filename in METRICS:
        fig, ax = plt.subplots(figsize=(8, 6))
        plotted = False

        for backend, rows in datasets.items():
            ok_rows = [r for r in rows if r.get("status") == "ok"]
            points = [
                (to_float(r["params_b"]), to_float(r.get(metric_key)), r.get("model"))
                for r in ok_rows
            ]
            points = [(x, y, m) for x, y, m in points if x is not None and y is not None]
            if not points:
                continue
            points.sort(key=lambda p: p[0])

            style = BACKEND_STYLE.get(backend, dict(color=None, marker="d", label=backend))
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            all_x_values.update(xs)

            ax.plot(xs, ys, marker=style["marker"], color=style["color"], label=style["label"], linewidth=2)
            for x, y, model_label in points:
                ax.annotate(
                    model_label, (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color=style["color"],
                )
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xscale("log")
        tick_values = sorted(all_x_values)
        ax.xaxis.set_major_locator(FixedLocator(tick_values))
        ax.xaxis.set_minor_locator(FixedLocator([]))  # no minor-tick clutter between points
        ax.xaxis.set_major_formatter(FixedFormatter([
            str(int(v)) if v == int(v) else str(v) for v in tick_values
        ]))

        ax.set_xlabel("Model size (billion params)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. model size")
        ax.grid(True, which="major", alpha=0.3)
        ax.legend()
        fig.tight_layout()

        out_path = PLOTS_DIR / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()

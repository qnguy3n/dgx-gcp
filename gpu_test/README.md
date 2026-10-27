# gpu_test

Standardized GPU inference benchmark: same model family (Qwen2.5) at 5
sizes from 1.5B to 72B, all 4-bit quantized, run on two backends (vLLM and
Ollama) on this DGX Spark (GB10). Produces one CSV per backend and
model-size-vs-metric plots for direct comparison.

## Why Qwen2.5 across all sizes

Using one model family end to end (rather than a different architecture at
each size point) isolates "model size" as the only variable that changes
between rows -- mixing architectures would confound the results. Both
backends run 4-bit weights for the same reason: vLLM uses Qwen's official
AWQ checkpoints, Ollama uses its default library tag (Q4_K_M for all five,
confirmed against the library before writing `models.py`).

| Model | Params | vLLM repo (AWQ) | Ollama tag (Q4_K_M) |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | `Qwen/Qwen2.5-1.5B-Instruct-AWQ` | `qwen2.5:1.5b` |
| Qwen2.5-7B-Instruct | 7B | `Qwen/Qwen2.5-7B-Instruct-AWQ` | `qwen2.5:7b` |
| Qwen2.5-14B-Instruct | 14B | `Qwen/Qwen2.5-14B-Instruct-AWQ` | `qwen2.5:14b` |
| Qwen2.5-32B-Instruct | 32B | `Qwen/Qwen2.5-32B-Instruct-AWQ` | `qwen2.5:32b` |
| Qwen2.5-72B-Instruct | 72B | `Qwen/Qwen2.5-72B-Instruct-AWQ` | `qwen2.5:72b` |

## Metrics measured (`bench_client.py`)

The standard LLM-serving benchmark set, all measured through the exact same
client code against each backend's OpenAI-compatible `/v1/chat/completions`
endpoint (vLLM natively, Ollama in its OpenAI-compat mode) -- so a metric
can never differ between backends because of a difference in how it was
measured, only because of real backend/model behavior:

- **TTFT** (`ttft_ms_mean`/`p95`) -- time to first token
- **ITL** (`itl_ms_mean`) -- inter-token latency during decode
- **Decode throughput** (`decode_tok_s_mean`) -- single request, unloaded GPU
- **Prefill throughput** (`prefill_tok_s_mean`) -- long prompt (~3000 tok), `max_tokens=1`, isolates prompt processing from decode
- **Concurrent throughput** (`concurrent_throughput_tok_s`) -- aggregate tok/s at concurrency=8
- **Concurrent latency** (`concurrent_mean_latency_ms`) -- per-request latency under that load
- **Load time** (`load_time_s`) -- cold start to first successful response
- **GPU memory footprint** (`model_mem_gib`) -- read from each backend's own reporting, see note below
- **GPU utilization / power draw** (`gpu_util_pct_mean/max`, `power_w_mean/max`) -- sampled every 0.3s during the benchmark window (`gpu_monitor.py`)

**Note on GPU memory**: `nvidia-smi --query-gpu=memory.used,memory.total`
reports `[N/A]` on this machine's driver (confirmed empirically, not a bug
here) -- so `gpu_monitor.py` deliberately does not query GPU memory.
`model_mem_gib` instead comes from vLLM's own startup log line
(`Model loading took X GiB memory`) and Ollama's `ollama ps` output.

## Files

| File | Role |
|---|---|
| `models.py` | The 5-model sweep definition |
| `bench_client.py` | Shared benchmark client (backend-agnostic) |
| `gpu_monitor.py` | Background GPU utilization/power sampler |
| `run_vllm_sweep.py` | Orchestrates the sweep on vLLM (starts/stops docker containers) |
| `run_ollama_sweep.py` | Orchestrates the sweep on Ollama (`ollama pull`/`stop`) |
| `plot_results.py` | Reads both CSVs, writes one PNG per metric to `results/plots/` |
| `results/vllm_results.csv`, `results/ollama_results.csv` | Raw per-model results |
| `results/logs/` | Per-model server/pull logs |

## Running

```bash
cd gpu_test
source .venv/bin/activate   # matplotlib + requests

# vLLM: stops the production server for the duration, restores it after
python3 run_vllm_sweep.py                                    # all 5 models
python3 run_vllm_sweep.py --models qwen2.5-1.5b,qwen2.5-7b    # subset, e.g. to validate first

# Ollama: requires Ollama installed and its server running first
python3 run_ollama_sweep.py
python3 run_ollama_sweep.py --models qwen2.5-1.5b,qwen2.5-7b

python3 plot_results.py     # safe to run after a partial sweep, or after each backend
```

## Resource cost -- read before running the full sweep

This is a genuinely large run, not a quick test:
- **Downloads**: ~75GB per backend (~150GB total) across the 5 sizes, dominated by the 72B model (~35-40GB per backend).
- **Time**: likely multiple hours end to end -- the 72B model alone can take a long time to download, load, and compile on first run.
- **GPU exclusivity**: only one large model fits in GPU memory at a time on this machine (confirmed empirically), so the vLLM sweep stops the production `gpt-oss-vllm` server for its entire duration.

Recommended: validate the pipeline on the two smallest models first
(`--models qwen2.5-1.5b,qwen2.5-7b`, a few minutes each) before committing
to the full 5-model run.

## Known gaps

- Ollama install is not automated here -- see the official installer
  (`curl -fsSL https://ollama.com/install.sh | sh`) and make sure it can
  see the GPU (`ollama ps` should show GPU processor usage, not CPU-only,
  once a model is loaded).
- `run_vllm_sweep.py` uses `vllm/vllm-openai:latest` with `--quantization awq`
  for every model -- if a given size's AWQ checkpoint ever moves/renames on
  HF, that row will fail with `status=failed_to_start`; the sweep logs the
  error and continues to the next model rather than aborting.

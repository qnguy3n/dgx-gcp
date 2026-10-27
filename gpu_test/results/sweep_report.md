# GPU throughput benchmark results

Qwen2.5 family, 4-bit quantized, 5 sizes (1.5B-72B). Source: results/vllm_results.csv, results/ollama_results.csv, results/ollama_tuned_results.csv, results/ollama_t4_results.csv. Plots: results/plots/decode_throughput.png, results/plots/concurrent_throughput.png.

## GPU VRAM required (GiB)

| Model | Params (B) | vLLM (AWQ, DGX) | Ollama (Q4_K_M, DGX) | Ollama tuned (Q4_K_M, DGX) | Ollama (Q4_K_M, T4) |
|---|---|---|---|---|---|
| qwen2.5-1.5b | 1.5 | 1.1 | 2.1 | 2.1 | 1.2 |
| qwen2.5-7b | 7 | 5.3 | 6.6 | 6.5 | 4.7 |
| qwen2.5-14b | 14 | 9.4 | 15.0 | 15.0 | 9.5 |
| qwen2.5-32b | 32 | 18.1 | 20.0 | 28.0 | 21.0 |
| qwen2.5-72b | 72 | 38.8 | 57.0 | 57.0 | — |

Resident model memory as reported by each backend after load (weights + runtime overhead), captured right after a single 1-token warm-up request -- **before** the concurrency=8 benchmark runs, so these numbers are not a live measurement under concurrent load.

**Why concurrency=8 doesn't cost ~8x VRAM:** weights are loaded once and shared by every concurrent request in a batch -- they aren't duplicated per request. Only the KV cache (cached attention keys/values) scales with concurrency, and it's small relative to weights, especially with grouped-query attention (few KV heads vs. many query heads). This box's `ollama.service` sets `OLLAMA_NUM_PARALLEL=8` and `OLLAMA_CONTEXT_LENGTH=4096`, so Ollama pre-reserves KV-cache space for 8 parallel slots x 4096 tokens at load time regardless of live traffic. For qwen2.5-32b (64 layers, 8 KV heads, head_dim 128, fp16 cache): `2 x 64 x 8 x 128 x 2 bytes = 256 KiB/token`, and `256 KiB x 4096 ctx x 8 slots ~= 8 GiB` -- which lines up almost exactly with the 20.0 -> 28.0 GiB jump between the Ollama (DGX) and Ollama tuned (DGX) columns above. Caveat: the same math predicts a comparable ~1.75 GiB bump for qwen2.5-7b, which wasn't observed, so this is a suggestive match rather than a fully controlled result -- the two runs were captured a day apart, not as an A/B, so measurement noise or per-model automatic sizing by Ollama can't be ruled out.

## T4 GPUs needed for a VRAM-fair concurrency=8 comparison

The T4 column above ran on a single 16 GB T4 and couldn't even load qwen2.5-32b/72b, so it isn't a fair comparison at concurrency=8 -- here's how many T4s it would take to just hold what the tuned DGX run needed (weights + the `OLLAMA_NUM_PARALLEL=8`/`OLLAMA_CONTEXT_LENGTH=4096` KV-cache reservation).

| Model | VRAM needed @ conc=8 (GiB) | T4s, bare minimum | T4s, recommended (+25% headroom) |
|---|---|---|---|
| qwen2.5-1.5b | 2.1 | 1 | 1 |
| qwen2.5-7b | 6.5 | 1 | 1 |
| qwen2.5-14b | 15.0 | 2 | 2 |
| qwen2.5-32b | 28.0 | 2 | 3 |
| qwen2.5-72b | 57.0 | 5 | 6 |

A T4's 16 GB nameplate reports as ~15 GiB in `nvidia-smi`, and CUDA context/driver overhead takes another slice, so this assumes **14 GiB usable per card**. "Bare minimum" is `ceil(VRAM needed / 14)` -- it lands models like qwen2.5-32b exactly on a card boundary with zero slack for the transient activation memory a concurrency=8 burst allocates on top of the resident footprint, so it'll likely OOM in practice. "Recommended" adds 25% headroom for that. Also: T4s have no NVLink, so pooling VRAM across cards for a model that doesn't fit on one relies on Ollama's own multi-GPU layer-splitting (pipeline-style, not tensor-parallel) -- it recovers the *capacity*, but cross-card PCIe hops per token mean decode latency still won't match the single-package DGX Spark's unified memory, even with "enough" T4s.

## Is a 4xT4 cloud box (e.g. GCP n1-standard-32) a fair DGX Spark comparison?

Checked against a concrete example: a GCP `n1-standard-32` (32 vCPUs, 120 GB host RAM) with 4xT4 attached -- 4 is also the max T4 count Google allows on a single VM (1, 2, or 4 only, never more). **VRAM fit:** 4 cards x 14 GiB usable = ~56 GiB covers 1.5b/7b/14b/32b with headroom to spare, but falls *short* of qwen2.5-72b's 57 GiB need -- i.e. the largest single-VM T4 config still can't fairly hold the biggest model in this sweep, let alone with the recommended 6-card margin.

Even where VRAM fits, three confounds ride along with swapping in T4s, so this isn't a controlled, single-variable comparison:

1. **GPU generation.** T4 is Turing (2018): 65 TFLOPS FP16, 320 GB/s GDDR6, 75 W, no native FP8/FP4 tensor engine. DGX Spark's GB10 is Blackwell (2025): ~11-12 TFLOPS FP16 measured but up to 1 PFLOP FP4 sparse, 273 GB/s unified memory. Opposite design points -- 4x the older card's dense FP16 doesn't translate to 4x the throughput that matters for a Q4_K_M-quantized workload.
2. **No NVLink.** T4s only talk over PCIe Gen3, so splitting 32b/72b across cards adds a cross-card hop per layer that DGX Spark's single coherent-memory package never pays -- decode latency won't be comparable even at matched VRAM.
3. **Memory model.** The 120 GB host RAM on the VM is separate DRAM, not usable as extra VRAM -- each T4's 16 GB is a hard wall. DGX Spark's 128 GB is one coherent pool CPU and GPU both draw from.

**Verdict:** legitimate as a "what would I actually rent instead" check, and it confirms 1.5b-32b run at all on cheaper/older hardware -- but it isn't a fair, single-variable hardware comparison, and it can't even host 72b with headroom. A cleaner test would size a *single* current-gen GPU with >=80 GB (e.g. an A100/H100 80GB or L40S) to hold 72b alone, isolating "GPU generation + unified vs. discrete memory" as the one thing that changes.

## Single-request decode throughput (tok/s)

| Model | Params (B) | vLLM (AWQ, DGX) | Ollama (Q4_K_M, DGX) | Ollama tuned (Q4_K_M, DGX) | Ollama (Q4_K_M, T4) |
|---|---|---|---|---|---|
| qwen2.5-1.5b | 1.5 | 124.7 (0.93x) | 158.4 (1.19x) | 162.9 (1.22x) | 133.7 |
| qwen2.5-7b | 7 | 39.6 (1.11x) | 34.9 (0.98x) | 36.9 (1.04x) | 35.6 |
| qwen2.5-14b | 14 | 21.7 (1.46x) | 19.6 (1.32x) | 21.2 (1.42x) | 14.9 |
| qwen2.5-32b | 32 | 11.4 (3.41x) | 9.0 (2.68x) | 6.8 (2.04x) | 3.3 |
| qwen2.5-72b | 72 | 5.4 | 4.3 | 4.2 | — |

Values in parentheses are the ratio vs. the Ollama (Q4_K_M, T4) column (e.g. 2.00x = twice the T4 throughput).

## Concurrent throughput, concurrency=8 (tok/s)

| Model | Params (B) | vLLM (AWQ, DGX) | Ollama (Q4_K_M, DGX) | Ollama tuned (Q4_K_M, DGX) | Ollama (Q4_K_M, T4) |
|---|---|---|---|---|---|
| qwen2.5-1.5b | 1.5 | 1119.7 (8.17x) | 162.4 (1.18x) | 807.4 (5.89x) | 137.1 |
| qwen2.5-7b | 7 | 331.7 (10.10x) | 35.4 (1.08x) | 168.9 (5.14x) | 32.9 |
| qwen2.5-14b | 14 | 178.2 (11.88x) | 19.4 (1.30x) | 108.5 (7.23x) | 15.0 |
| qwen2.5-32b | 32 | 89.5 (26.65x) | 8.9 (2.66x) | 42.4 (12.63x) | 3.4 |
| qwen2.5-72b | 72 | 41.1 | 4.2 | 27.3 | — |

Values in parentheses are the ratio vs. the Ollama (Q4_K_M, T4) column (e.g. 2.00x = twice the T4 throughput).

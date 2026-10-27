# GPU throughput benchmark results

Qwen2.5 family, 4-bit quantized, 5 sizes (1.5B-72B). Source: results/vllm_results.csv, results/ollama_results.csv, results/ollama_t4_results.csv. Plots: results/plots/decode_throughput.png, results/plots/concurrent_throughput.png.

## Single-request decode throughput (tok/s)

| Model | Params (B) | vLLM (AWQ, DGX) | Ollama (Q4_K_M, DGX) | Ollama (Q4_K_M, T4) |
|---|---|---|---|---|
| qwen2.5-1.5b | 1.5 | 124.7 (0.93x) | 158.4 (1.19x) | 133.7 |
| qwen2.5-7b | 7 | 39.6 (1.11x) | 34.9 (0.98x) | 35.6 |
| qwen2.5-14b | 14 | 21.7 (1.46x) | 19.6 (1.32x) | 14.9 |
| qwen2.5-32b | 32 | 11.4 (3.41x) | 9.0 (2.68x) | 3.3 |
| qwen2.5-72b | 72 | 5.4 | 4.3 | — |

Values in parentheses are the ratio vs. the Ollama (Q4_K_M, T4) column (e.g. 2.00x = twice the T4 throughput).

## Concurrent throughput, concurrency=8 (tok/s)

| Model | Params (B) | vLLM (AWQ, DGX) | Ollama (Q4_K_M, DGX) | Ollama (Q4_K_M, T4) |
|---|---|---|---|---|
| qwen2.5-1.5b | 1.5 | 1119.7 (8.17x) | 162.4 (1.18x) | 137.1 |
| qwen2.5-7b | 7 | 331.7 (10.10x) | 35.4 (1.08x) | 32.9 |
| qwen2.5-14b | 14 | 178.2 (11.88x) | 19.4 (1.30x) | 15.0 |
| qwen2.5-32b | 32 | 89.5 (26.65x) | 8.9 (2.66x) | 3.4 |
| qwen2.5-72b | 72 | 41.1 | 4.2 | — |

Values in parentheses are the ratio vs. the Ollama (Q4_K_M, T4) column (e.g. 2.00x = twice the T4 throughput).

"""Shared benchmark client for any OpenAI-compatible /v1/chat/completions
endpoint. vLLM and Ollama (in its OpenAI-compat mode) both speak this
protocol, so this exact same code measures both backends -- backend
differences live only in the sweep scripts that start/stop each server,
never in how a metric is measured.

Metrics (the standard LLM-serving benchmark set):
  ttft_ms            time to first token
  itl_ms             inter-token latency during decode (ms/token)
  decode_tok_s        single-request decode throughput (unloaded GPU)
  prefill_tok_s        prompt-processing throughput (long prompt, max_tokens=1)
  concurrent_throughput_tok_s   aggregate tok/s with N simultaneous requests
  concurrent_mean_latency_ms     per-request latency under that load
"""
from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

SHORT_PROMPT = (
    "Summarize, in one sentence, why the sky appears blue during the day "
    "and orange/red at sunset. Then list three other everyday phenomena "
    "caused by light scattering."
)
# ~3000 tokens of filler + a trivial question. Used only with max_tokens=1
# to isolate prompt-processing (prefill) speed from decode speed.
LONG_PROMPT = ("The quick brown fox jumps over the lazy dog. " * 300) + \
    "\n\nGiven the text above, respond with just the word 'done'."


def _post(base_url: str, model: str, messages: list, max_tokens: int, stream: bool):
    return requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json={"model": model, "messages": messages, "max_tokens": max_tokens,
              "temperature": 0, "stream": stream},
        stream=stream, timeout=600,
    )


def _stream_once(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    messages = [{"role": "user", "content": prompt}]
    t0 = time.monotonic()
    resp = _post(base_url, model, messages, max_tokens, stream=True)
    resp.raise_for_status()

    token_times = []
    first_token_time = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or [{}]
        delta = choices[0].get("delta", {})
        if delta.get("content") or delta.get("reasoning"):
            now = time.monotonic()
            if first_token_time is None:
                first_token_time = now
            token_times.append(now)

    total_end = time.monotonic()
    if not token_times:
        return {"ttft_ms": None, "total_ms": (total_end - t0) * 1000, "n_tokens": 0, "itl_ms": None}

    ttft_ms = (first_token_time - t0) * 1000
    total_ms = (total_end - t0) * 1000
    n_tokens = len(token_times)
    itl_ms = None
    if n_tokens > 1:
        gaps = [(token_times[i] - token_times[i - 1]) * 1000 for i in range(1, n_tokens)]
        itl_ms = sum(gaps) / len(gaps)
    return {"ttft_ms": ttft_ms, "total_ms": total_ms, "n_tokens": n_tokens, "itl_ms": itl_ms}


def measure_single_stream(base_url: str, model: str, reps: int = 5, max_tokens: int = 256) -> dict:
    runs = [_stream_once(base_url, model, SHORT_PROMPT, max_tokens) for _ in range(reps)]
    runs = [r for r in runs if r["n_tokens"] > 0]
    if not runs:
        return {"ttft_ms_mean": None, "ttft_ms_p95": None, "itl_ms_mean": None, "decode_tok_s_mean": None}
    ttfts = sorted(r["ttft_ms"] for r in runs)
    itls = [r["itl_ms"] for r in runs if r["itl_ms"] is not None]
    decode_tps = [r["n_tokens"] / (r["total_ms"] / 1000) for r in runs]
    return {
        "ttft_ms_mean": round(statistics.mean(ttfts), 1),
        "ttft_ms_p95": round(ttfts[min(int(0.95 * (len(ttfts) - 1)), len(ttfts) - 1)], 1),
        "itl_ms_mean": round(statistics.mean(itls), 2) if itls else None,
        "decode_tok_s_mean": round(statistics.mean(decode_tps), 2),
    }


def measure_prefill(base_url: str, model: str, reps: int = 3) -> dict:
    speeds = []
    prompt_tokens = None
    for _ in range(reps):
        t0 = time.monotonic()
        resp = _post(base_url, model, [{"role": "user", "content": LONG_PROMPT}], 1, stream=False)
        resp.raise_for_status()
        elapsed = time.monotonic() - t0
        usage = (resp.json().get("usage")) or {}
        pt = usage.get("prompt_tokens")
        if pt:
            prompt_tokens = pt
            speeds.append(pt / elapsed)
    if not speeds:
        return {"prefill_tok_s_mean": None, "prompt_tokens": None}
    return {"prefill_tok_s_mean": round(statistics.mean(speeds), 1), "prompt_tokens": prompt_tokens}


def _one_concurrent_request(base_url: str, model: str, max_tokens: int) -> dict:
    t0 = time.monotonic()
    resp = _post(base_url, model, [{"role": "user", "content": SHORT_PROMPT}], max_tokens, stream=False)
    resp.raise_for_status()
    elapsed = time.monotonic() - t0
    usage = (resp.json().get("usage")) or {}
    return {"latency_ms": elapsed * 1000, "completion_tokens": usage.get("completion_tokens", 0)}


def measure_concurrent(base_url: str, model: str, concurrency: int = 8, max_tokens: int = 256) -> dict:
    wall_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_one_concurrent_request, base_url, model, max_tokens) for _ in range(concurrency)]
        results = [f.result() for f in as_completed(futures)]
    wall_s = time.monotonic() - wall_start
    total_tokens = sum(r["completion_tokens"] for r in results)
    latencies = [r["latency_ms"] for r in results]
    return {
        "concurrency": concurrency,
        "concurrent_throughput_tok_s": round(total_tokens / wall_s, 2) if wall_s else None,
        "concurrent_mean_latency_ms": round(statistics.mean(latencies), 1),
    }


def run_full_benchmark(base_url: str, model: str, concurrency: int = 8) -> dict:
    """Runs the full standard battery and returns one flat dict of metrics."""
    result = {}
    result.update(measure_single_stream(base_url, model))
    result.update(measure_prefill(base_url, model))
    result.update(measure_concurrent(base_url, model, concurrency=concurrency))
    return result

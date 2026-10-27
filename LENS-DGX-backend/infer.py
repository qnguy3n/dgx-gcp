"""Calls the local vLLM OpenAI-compatible server."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests

from config import Config


class TransientInferenceError(Exception):
    """Retryable: connection refused, timeout, 5xx. Caller should nack the message."""


class PermanentInferenceError(Exception):
    """Not retryable: 4xx, malformed response. Caller should ack with an error row."""


@dataclass
class InferenceResult:
    content: str
    reasoning: Optional[str]
    latency_ms: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]


def run_inference(text: str, config: Config) -> InferenceResult:
    """Send `text` verbatim as the user turn.

    The instruction is already embedded in the message body (per the
    producer's contract), so no system prompt is added here.
    """
    url = f"{config.vllm_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": config.model_name,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": config.max_output_tokens,
    }

    start = time.monotonic()
    try:
        resp = requests.post(url, json=payload, timeout=config.request_timeout_s)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise TransientInferenceError(f"vLLM request failed: {exc}") from exc
    latency_ms = (time.monotonic() - start) * 1000

    if resp.status_code >= 500:
        raise TransientInferenceError(f"vLLM returned {resp.status_code}: {resp.text[:500]}")
    if resp.status_code >= 400:
        raise PermanentInferenceError(f"vLLM returned {resp.status_code}: {resp.text[:500]}")

    try:
        data = resp.json()
        message = data["choices"][0]["message"]
        content = message["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise PermanentInferenceError(f"Unexpected vLLM response shape: {exc}") from exc

    usage = data.get("usage") or {}
    return InferenceResult(
        content=content,
        reasoning=message.get("reasoning"),
        latency_ms=latency_ms,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )

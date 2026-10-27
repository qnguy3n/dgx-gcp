"""The model sweep.

Deliberately one model family (Qwen2.5) across sizes, so model size is the
only variable that changes between rows -- mixing architectures would
confound the "does size predict speed" question this test is asking.
Both backends run 4-bit weights: vLLM uses Qwen's official AWQ checkpoints,
Ollama uses its default library tag (Q4_K_M for all five of these,
confirmed against the Ollama library page before writing this list).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    label: str        # short id used in filenames/logs/CSV rows, e.g. "qwen2.5-7b"
    params_b: float    # parameter count in billions -- the x-axis for plots
    vllm_repo: str      # HF repo id, AWQ-quantized (verified to exist on HF)
    ollama_tag: str      # Ollama library tag (verified to exist in the library)


MODELS = [
    ModelSpec("qwen2.5-1.5b", 1.5, "Qwen/Qwen2.5-1.5B-Instruct-AWQ", "qwen2.5:1.5b"),
    ModelSpec("qwen2.5-7b", 7, "Qwen/Qwen2.5-7B-Instruct-AWQ", "qwen2.5:7b"),
    ModelSpec("qwen2.5-14b", 14, "Qwen/Qwen2.5-14B-Instruct-AWQ", "qwen2.5:14b"),
    ModelSpec("qwen2.5-32b", 32, "Qwen/Qwen2.5-32B-Instruct-AWQ", "qwen2.5:32b"),
    ModelSpec("qwen2.5-72b", 72, "Qwen/Qwen2.5-72B-Instruct-AWQ", "qwen2.5:72b"),
]

MODELS_BY_LABEL = {m.label: m for m in MODELS}

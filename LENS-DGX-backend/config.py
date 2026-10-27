"""Configuration for the LENS DGX backend: Pub/Sub -> vLLM -> BigQuery.

All settings come from environment variables so dev/prod can be selected
without touching code. Set LENS_DGX_ENV=dev (default) or LENS_DGX_ENV=prod.

Fully self-contained: credentials live in ./credentials/, no reach-through
to any sibling folder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_CREDENTIALS_DIR = os.path.join(_HERE, "credentials")


def _env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Config:
    env: str
    project_id: str
    subscription_id: str
    credentials_path: str
    bq_table_id: str
    vllm_url: str
    model_name: str
    max_concurrency: int
    request_timeout_s: float
    max_output_tokens: int
    log_dir: str


def load_config() -> Config:
    env = (_env("LENS_DGX_ENV", "dev") or "dev").lower()

    if env == "dev":
        project_id = _env("LENS_DGX_PROJECT_ID", "ml-model-runner-f22694eb")
        subscription_id = _env("LENS_DGX_SUBSCRIPTION_ID", "lens-dgx-spark-dev-input")
        credentials_path = _env(
            "GOOGLE_APPLICATION_CREDENTIALS",
            os.path.join(_CREDENTIALS_DIR, "dev_auth.json"),
        )
        bq_table_id = _env(
            "LENS_DGX_BQ_TABLE_ID",
            "ml-model-runner-f22694eb.lens_dgx_spark_dev.results",
        )
    elif env == "prod":
        # No prod subscription/table id has been confirmed yet -- require
        # them explicitly rather than guessing.
        project_id = _env("LENS_DGX_PROJECT_ID", required=True)
        subscription_id = _env("LENS_DGX_SUBSCRIPTION_ID", required=True)
        credentials_path = _env(
            "GOOGLE_APPLICATION_CREDENTIALS",
            os.path.join(_CREDENTIALS_DIR, "prod_auth.json"),
        )
        bq_table_id = _env("LENS_DGX_BQ_TABLE_ID", required=True)
    else:
        raise RuntimeError(f"Unknown LENS_DGX_ENV={env!r}, expected 'dev' or 'prod'")

    return Config(
        env=env,
        project_id=project_id,
        subscription_id=subscription_id,
        credentials_path=credentials_path,
        bq_table_id=bq_table_id,
        vllm_url=_env("LENS_DGX_VLLM_URL", "http://localhost:8000"),
        model_name=_env("LENS_DGX_MODEL_NAME", "openai/gpt-oss-20b"),
        max_concurrency=int(_env("LENS_DGX_MAX_CONCURRENCY", "4")),
        request_timeout_s=float(_env("LENS_DGX_REQUEST_TIMEOUT_S", "120")),
        max_output_tokens=int(_env("LENS_DGX_MAX_OUTPUT_TOKENS", "2048")),
        log_dir=_env("LENS_DGX_LOG_DIR", os.path.join(_HERE, "logs")),
    )

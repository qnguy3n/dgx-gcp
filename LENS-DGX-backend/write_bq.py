"""Builds and inserts result rows into BigQuery.

Schema (see create_table.py):
    run_id       STRING    REQUIRED  -- the only column guaranteed to exist
    row_index    INTEGER             -- position within this message's CSV output
    fields       JSON                -- parsed CSV row as key/value; columns vary by run
    raw_row      STRING              -- original CSV line (or full raw output on parse_error)
    status       STRING              -- "ok" | "parse_error" | "inference_error" | "bad_message"
    error        STRING
    model_name   STRING
    latency_ms   FLOAT               -- vLLM inference time for this message
    processed_at TIMESTAMP
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from google.cloud import bigquery


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_ok_rows(
    run_id: str,
    parsed_rows: List[Dict[str, str]],
    model_name: str,
    latency_ms: float,
) -> List[dict]:
    now = _now_iso()
    return [
        {
            "run_id": run_id,
            "row_index": i,
            # JSON-typed columns go over the legacy streaming-insert API
            # (insert_rows_json) as a JSON *string*, not a nested dict --
            # passing a dict here gets misread as an attempt to populate a
            # RECORD column and is rejected with "not a record".
            "fields": json.dumps(row),
            # str()-coerce defensively: csv_parse.py's contract is
            # Dict[str, str], but a crash here previously took the whole
            # service down (see PROGRESS_LOG.md, 2026-10-28) when that
            # contract was violated. Cheap insurance against it happening
            # again from some other code path.
            "raw_row": ",".join("" if v is None else str(v) for v in row.values()),
            "status": "ok",
            "error": None,
            "model_name": model_name,
            "latency_ms": latency_ms,
            "processed_at": now,
        }
        for i, row in enumerate(parsed_rows)
    ]


def build_error_row(
    run_id: str,
    status: str,
    error: str,
    raw_output: Optional[str] = None,
    model_name: Optional[str] = None,
    latency_ms: Optional[float] = None,
) -> dict:
    return {
        "run_id": run_id,
        "row_index": 0,
        "fields": None,
        "raw_row": raw_output,
        "status": status,
        "error": error[:2000],
        "model_name": model_name,
        "latency_ms": latency_ms,
        "processed_at": _now_iso(),
    }


def insert_rows(client: bigquery.Client, table_id: str, rows: List[dict]) -> None:
    if not rows:
        return
    # row_ids give BigQuery's streaming insert best-effort dedup, so a
    # Pub/Sub redelivery of the same message doesn't double up rows.
    row_ids = [f"{r['run_id']}:{r['row_index']}" for r in rows]
    errors = client.insert_rows_json(table_id, rows, row_ids=row_ids)
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")

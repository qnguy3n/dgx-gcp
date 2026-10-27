"""Structured per-message logging for testing/debugging.

Writes one JSON line per processed message to LENS_DGX_LOG_DIR/inference_log.jsonl
with the input, the model's raw output, the outcome, and timing -- separate
from the regular application log (which stays human-readable). Meant to be
tailed/grepped while testing the pipeline end to end, e.g.:

    tail -f service/logs/inference_log.jsonl | jq .
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Optional


class InferenceLogger:
    def __init__(self, log_dir: str, filename: str = "inference_log.jsonl") -> None:
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, filename)
        self._lock = threading.Lock()

    def log(
        self,
        *,
        message_id: str,
        run_id: str,
        input_text: str,
        status: str,
        output_text: Optional[str] = None,
        error: Optional[str] = None,
        project_id: Optional[str] = None,
        letter_id: Optional[str] = None,
        datetime_sent: Optional[str] = None,
        received_at: Optional[str] = None,
        transit_latency_ms: Optional[float] = None,
        inference_latency_ms: Optional[float] = None,
        bq_insert_latency_ms: Optional[float] = None,
        total_processing_ms: Optional[float] = None,
        row_count: Optional[int] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_id": message_id,
            "run_id": run_id,
            "project_id": project_id,
            "letter_id": letter_id,
            "status": status,
            "input_preview": input_text[:500] if input_text else None,
            "input_chars": len(input_text) if input_text else 0,
            "output_preview": output_text[:1000] if output_text else None,
            "row_count": row_count,
            "error": error,
            # Producer-reported send time vs. our own wall-clock receipt time,
            # so transit_latency_ms covers Pub/Sub delivery, not just our own work.
            "datetime_sent": datetime_sent,
            "received_at": received_at,
            "transit_latency_ms": round(transit_latency_ms, 1) if transit_latency_ms is not None else None,
            "inference_latency_ms": round(inference_latency_ms, 1) if inference_latency_ms is not None else None,
            "bq_insert_latency_ms": round(bq_insert_latency_ms, 1) if bq_insert_latency_ms is not None else None,
            "total_processing_ms": round(total_processing_ms, 1) if total_processing_ms is not None else None,
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

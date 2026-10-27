"""Pub/Sub message handling: parse -> infer -> parse CSV -> write to BigQuery -> ack/nack.

Every message also gets a structured entry in inference_log.jsonl (see
test_logger.py) recording the input, the model's output, the outcome, and
timing -- for testing/debugging the pipeline end to end.
"""
from __future__ import annotations

import json
import logging
import time

from google.cloud import bigquery, pubsub_v1

from config import Config
from csv_parse import parse_csv_output
from infer import PermanentInferenceError, TransientInferenceError, run_inference
from test_logger import InferenceLogger
from write_bq import build_error_row, build_ok_rows, insert_rows

logger = logging.getLogger("codcot.consume")

# The producer's message shape is {"run_id": ..., "letter_body": "..."} per
# push.py; accept a couple of alternate key names defensively.
_TEXT_KEYS = ("letter_body", "text", "prompt")


def _extract_text(payload: dict) -> str:
    for key in _TEXT_KEYS:
        if key in payload and payload[key]:
            return payload[key]
    raise ValueError(f"no text field found (looked for {_TEXT_KEYS}) in {list(payload)}")


def make_callback(config: Config, bq_client: bigquery.Client, inference_log: InferenceLogger):
    def callback(message: pubsub_v1.subscriber.message.Message) -> None:
        t_start = time.monotonic()
        message_id = message.message_id

        try:
            payload = json.loads(message.data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("message %s: bad JSON, acking and recording: %s", message_id, exc)
            run_id = f"unknown:{message_id}"
            insert_rows(bq_client, config.bq_table_id, [
                build_error_row(run_id=run_id, status="bad_message", error=str(exc)),
            ])
            inference_log.log(
                message_id=message_id, run_id=run_id, input_text="", status="bad_message",
                error=str(exc), total_processing_ms=(time.monotonic() - t_start) * 1000,
            )
            message.ack()
            return

        run_id = payload.get("run_id", message_id)

        try:
            text = _extract_text(payload)
        except ValueError as exc:
            logger.error("message %s (run_id=%s): %s", message_id, run_id, exc)
            insert_rows(bq_client, config.bq_table_id, [
                build_error_row(run_id=run_id, status="bad_message", error=str(exc)),
            ])
            inference_log.log(
                message_id=message_id, run_id=run_id, input_text="", status="bad_message",
                error=str(exc), total_processing_ms=(time.monotonic() - t_start) * 1000,
            )
            message.ack()
            return

        try:
            result = run_inference(text, config)
        except TransientInferenceError as exc:
            logger.warning("run_id=%s: transient inference error, nacking: %s", run_id, exc)
            inference_log.log(
                message_id=message_id, run_id=run_id, input_text=text, status="transient_error",
                error=str(exc), total_processing_ms=(time.monotonic() - t_start) * 1000,
            )
            message.nack()
            return
        except PermanentInferenceError as exc:
            logger.error("run_id=%s: permanent inference error: %s", run_id, exc)
            insert_rows(bq_client, config.bq_table_id, [
                build_error_row(run_id=run_id, status="inference_error", error=str(exc)),
            ])
            inference_log.log(
                message_id=message_id, run_id=run_id, input_text=text, status="inference_error",
                error=str(exc), total_processing_ms=(time.monotonic() - t_start) * 1000,
            )
            message.ack()
            return

        try:
            parsed_rows = parse_csv_output(result.content)
            rows = build_ok_rows(run_id, parsed_rows, config.model_name, result.latency_ms)
            log_status, log_error = "ok", None
        except ValueError as exc:
            logger.warning("run_id=%s: CSV parse failed: %s", run_id, exc)
            rows = [build_error_row(
                run_id=run_id, status="parse_error", error=str(exc),
                raw_output=result.content, model_name=config.model_name,
                latency_ms=result.latency_ms,
            )]
            log_status, log_error = "parse_error", str(exc)

        try:
            insert_rows(bq_client, config.bq_table_id, rows)
        except RuntimeError as exc:
            logger.error("run_id=%s: BigQuery insert failed, nacking for retry: %s", run_id, exc)
            inference_log.log(
                message_id=message_id, run_id=run_id, input_text=text, status="bq_insert_error",
                output_text=result.content, error=str(exc),
                inference_latency_ms=result.latency_ms,
                total_processing_ms=(time.monotonic() - t_start) * 1000,
            )
            message.nack()
            return

        total_ms = (time.monotonic() - t_start) * 1000
        logger.info(
            "run_id=%s: wrote %d row(s), status=%s, inference=%.0fms, total=%.0fms",
            run_id, len(rows), log_status, result.latency_ms, total_ms,
        )
        inference_log.log(
            message_id=message_id, run_id=run_id, input_text=text, status=log_status,
            output_text=result.content, error=log_error,
            inference_latency_ms=result.latency_ms, total_processing_ms=total_ms,
            row_count=len(rows),
        )
        message.ack()

    return callback

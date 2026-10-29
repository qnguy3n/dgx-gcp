"""Pub/Sub message handling: parse -> infer -> parse CSV -> write to BigQuery -> ack/nack.

Every message also gets a structured entry in inference_log.jsonl (see
test_logger.py) recording the input, the model's output, the outcome, and
timing -- for testing/debugging the pipeline end to end.

Timing breakdown recorded per message:
    transit_latency_ms   -- producer's datetime_sent -> our own wall-clock
                            receipt (received_at). Covers Pub/Sub delivery,
                            not our own processing.
    inference_latency_ms -- vLLM call only (see infer.py).
    bq_insert_latency_ms -- the BigQuery streaming-insert call only.
    total_processing_ms  -- full handling, receipt to ack/nack.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from google.cloud import bigquery, pubsub_v1

from config import Config
from csv_parse import parse_csv_output
from infer import PermanentInferenceError, TransientInferenceError, run_inference
from test_logger import InferenceLogger
from write_bq import build_error_row, build_ok_rows, insert_rows

logger = logging.getLogger("lens_dgx.consume")

# The producer's message shape is {"run_id": ..., "project_id": ...,
# "letter_id": ..., "datetime_sent": ..., "letter_body": "..."} per
# tools/publish_test_message.py; accept a couple of alternate text key names
# defensively, and treat project_id/letter_id/datetime_sent as optional so
# older-shaped messages still work.
_TEXT_KEYS = ("letter_body", "text", "prompt")


def _extract_text(payload: dict) -> str:
    for key in _TEXT_KEYS:
        if key in payload and payload[key]:
            return payload[key]
    raise ValueError(f"no text field found (looked for {_TEXT_KEYS}) in {list(payload)}")


def _transit_latency_ms(datetime_sent: Optional[str], received_at: datetime) -> Optional[float]:
    """How long the message spent between the producer sending it and us
    receiving it here (Pub/Sub delivery time), per the producer's own
    `datetime_sent` timestamp. None if missing/unparseable."""
    if not datetime_sent:
        return None
    try:
        sent_at = datetime.fromisoformat(str(datetime_sent).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (received_at - sent_at).total_seconds() * 1000


def _insert_rows_timed(bq_client: bigquery.Client, table_id: str, rows: list) -> float:
    """Runs insert_rows and returns elapsed ms. Propagates RuntimeError on failure
    (callers that care about elapsed time on a failed insert aren't timed here)."""
    t0 = time.monotonic()
    insert_rows(bq_client, table_id, rows)
    return (time.monotonic() - t0) * 1000


def make_callback(config: Config, bq_client: bigquery.Client, inference_log: InferenceLogger):
    def _process_message(message: pubsub_v1.subscriber.message.Message) -> None:
        received_at_dt = datetime.now(timezone.utc)
        received_at = received_at_dt.isoformat()
        t_start = time.monotonic()
        message_id = message.message_id

        try:
            payload = json.loads(message.data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("message %s: bad JSON, acking and recording: %s", message_id, exc)
            run_id = f"unknown:{message_id}"
            bq_insert_ms = _insert_rows_timed(bq_client, config.bq_table_id, [
                build_error_row(run_id=run_id, status="bad_message", error=str(exc)),
            ])
            inference_log.log(
                message_id=message_id, run_id=run_id, input_text="", status="bad_message",
                error=str(exc), received_at=received_at,
                bq_insert_latency_ms=bq_insert_ms,
                total_processing_ms=(time.monotonic() - t_start) * 1000,
            )
            message.ack()
            return

        run_id = payload.get("run_id", message_id)
        project_id = payload.get("project_id")
        letter_id = payload.get("letter_id")
        datetime_sent = payload.get("datetime_sent")
        transit_ms = _transit_latency_ms(datetime_sent, received_at_dt)
        if transit_ms is not None:
            logger.info(
                "run_id=%s (project_id=%s, letter_id=%s): received at %s, transit=%.0fms",
                run_id, project_id, letter_id, received_at, transit_ms,
            )

        try:
            text = _extract_text(payload)
        except ValueError as exc:
            logger.error("message %s (run_id=%s): %s", message_id, run_id, exc)
            bq_insert_ms = _insert_rows_timed(bq_client, config.bq_table_id, [
                build_error_row(run_id=run_id, status="bad_message", error=str(exc)),
            ])
            inference_log.log(
                message_id=message_id, run_id=run_id, input_text="", status="bad_message",
                error=str(exc), project_id=project_id, letter_id=letter_id,
                datetime_sent=datetime_sent, received_at=received_at,
                transit_latency_ms=transit_ms, bq_insert_latency_ms=bq_insert_ms,
                total_processing_ms=(time.monotonic() - t_start) * 1000,
            )
            message.ack()
            return

        try:
            result = run_inference(text, config)
        except TransientInferenceError as exc:
            logger.warning("run_id=%s: transient inference error, nacking: %s", run_id, exc)
            inference_log.log(
                message_id=message_id, run_id=run_id, input_text=text, status="transient_error",
                error=str(exc), project_id=project_id, letter_id=letter_id,
                datetime_sent=datetime_sent, received_at=received_at,
                transit_latency_ms=transit_ms,
                total_processing_ms=(time.monotonic() - t_start) * 1000,
            )
            message.nack()
            return
        except PermanentInferenceError as exc:
            logger.error("run_id=%s: permanent inference error: %s", run_id, exc)
            bq_insert_ms = _insert_rows_timed(bq_client, config.bq_table_id, [
                build_error_row(run_id=run_id, status="inference_error", error=str(exc)),
            ])
            inference_log.log(
                message_id=message_id, run_id=run_id, input_text=text, status="inference_error",
                error=str(exc), project_id=project_id, letter_id=letter_id,
                datetime_sent=datetime_sent, received_at=received_at,
                transit_latency_ms=transit_ms, bq_insert_latency_ms=bq_insert_ms,
                total_processing_ms=(time.monotonic() - t_start) * 1000,
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
            bq_insert_ms = _insert_rows_timed(bq_client, config.bq_table_id, rows)
        except RuntimeError as exc:
            logger.error("run_id=%s: BigQuery insert failed, nacking for retry: %s", run_id, exc)
            inference_log.log(
                message_id=message_id, run_id=run_id, input_text=text, status="bq_insert_error",
                output_text=result.content, error=str(exc),
                project_id=project_id, letter_id=letter_id,
                datetime_sent=datetime_sent, received_at=received_at,
                transit_latency_ms=transit_ms,
                inference_latency_ms=result.latency_ms,
                total_processing_ms=(time.monotonic() - t_start) * 1000,
            )
            message.nack()
            return

        total_ms = (time.monotonic() - t_start) * 1000
        logger.info(
            "run_id=%s: wrote %d row(s), status=%s, transit=%s, inference=%.0fms, bq_insert=%.0fms, total=%.0fms",
            run_id, len(rows), log_status,
            f"{transit_ms:.0f}ms" if transit_ms is not None else "n/a",
            result.latency_ms, bq_insert_ms, total_ms,
        )
        inference_log.log(
            message_id=message_id, run_id=run_id, input_text=text, status=log_status,
            output_text=result.content, error=log_error,
            project_id=project_id, letter_id=letter_id,
            datetime_sent=datetime_sent, received_at=received_at,
            transit_latency_ms=transit_ms,
            inference_latency_ms=result.latency_ms,
            bq_insert_latency_ms=bq_insert_ms,
            total_processing_ms=total_ms,
            row_count=len(rows),
        )
        message.ack()

    def callback(message: pubsub_v1.subscriber.message.Message) -> None:
        # Last-resort safety net. On 2026-10-28, a malformed/ragged CSV
        # row from the model produced a non-string dict value that
        # crashed _process_message with an uncaught TypeError; that
        # propagated out of the callback, through the Pub/Sub client's
        # streaming-pull future, and into main.py's `except Exception`
        # handler, which logs it as "stream ended" and returns 0 --
        # a *clean* exit, so systemd's Restart=on-failure never fired.
        # The whole service silently stopped processing all messages
        # until someone noticed. No single message should ever be able
        # to do that again: catch anything _process_message doesn't
        # already handle itself, log it, and nack for retry instead of
        # letting it escape.
        try:
            _process_message(message)
        except Exception:
            logger.exception(
                "message %s: unhandled exception in callback, nacking for retry", message.message_id,
            )
            message.nack()

    return callback

"""End-to-end test: publish a message to Pub/Sub, then poll BigQuery for the result.

Usage:
    python test_e2e_pubsub_bq.py [--timeout 120] [--letter-text "some text"]

What it does:
    1. Publishes one message to the Pub/Sub topic in LENS run_id format:
           run_id = "{lens_run_id}:{letter_id}"
    2. Polls the BigQuery results table every 5 s until a matching row appears
       (or the timeout elapses).
    3. Prints a summary of the result.

Credentials: uses dev_auth.json (same as other test scripts here).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# --- Credentials ---------------------------------------------------------
# Publishing requires user-level permissions (the dev SA lacks pubsub.topics.publish).
# Use Application Default Credentials (gcloud auth login) for both Pub/Sub and BQ.
# If GOOGLE_APPLICATION_CREDENTIALS is already set in the environment it will be used;
# otherwise ADC (gcloud user credentials) are picked up automatically.

PROJECT_ID = "ml-model-runner-f22694eb"
TOPIC_ID = "lens-dgx-spark-dev-input"
BQ_TABLE_ID = "ml-model-runner-f22694eb.lens_dgx_spark_dev.results"
POLL_INTERVAL_S = 5


def publish(letter_id: str, lens_run_id: str, letter_text: str) -> str:
    from google.cloud import pubsub_v1  # type: ignore

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)

    payload = {
        "run_id": f"{lens_run_id}:{letter_id}",
        "project_id": "test-study",
        "letter_id": letter_id,
        "datetime_sent": datetime.now(timezone.utc).isoformat(),
        "letter_body": letter_text,
    }
    data = json.dumps(payload).encode("utf-8")
    future = publisher.publish(topic_path, data)
    msg_id = future.result(timeout=10)
    return msg_id


def poll_bq(lens_run_id: str, timeout_s: float) -> dict | None:
    from google.cloud import bigquery  # type: ignore

    client = bigquery.Client(project=PROJECT_ID)
    prefix = f"{lens_run_id}:"
    query = f"""
        SELECT run_id, fields, raw_row, status, error, latency_ms
        FROM `{BQ_TABLE_ID}`
        WHERE STARTS_WITH(run_id, @prefix)
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("prefix", "STRING", prefix)]
    )

    deadline = time.monotonic() + timeout_s
    poll_num = 0
    while time.monotonic() < deadline:
        poll_num += 1
        rows = list(client.query(query, job_config=job_config).result())
        if rows:
            return dict(rows[0])
        remaining = deadline - time.monotonic()
        print(f"  [poll #{poll_num}] no result yet — waiting {POLL_INTERVAL_S}s "
              f"({remaining:.0f}s left)")
        time.sleep(POLL_INTERVAL_S)

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E test: Pub/Sub → DGX → BigQuery")
    parser.add_argument("--timeout", type=float, default=120,
                        help="Seconds to wait for BQ result (default: 120)")
    parser.add_argument("--letter-text", default=(
        "Patient presents with bilateral visual field loss. "
        "Diagnosed with retinitis pigmentosa. CMO noted in left eye. "
        "Follow-up in 3 months."
    ), help="Letter text to send as the prompt body")
    args = parser.parse_args()

    lens_run_id = "test-" + uuid.uuid4().hex[:8]
    letter_id = "letter-" + uuid.uuid4().hex[:6]

    print(f"lens_run_id : {lens_run_id}")
    print(f"letter_id   : {letter_id}")
    print(f"run_id field: {lens_run_id}:{letter_id}")
    print(f"BQ table    : {BQ_TABLE_ID}")
    print()

    # 1. Publish
    print("Publishing message to Pub/Sub …")
    try:
        msg_id = publish(letter_id, lens_run_id, args.letter_text)
        print(f"  Published. Pub/Sub message ID: {msg_id}")
    except Exception as exc:
        print(f"  FAILED to publish: {exc}", file=sys.stderr)
        sys.exit(1)

    # 2. Poll
    print(f"\nPolling BigQuery (timeout={args.timeout}s, interval={POLL_INTERVAL_S}s) …")
    start = time.monotonic()
    result = poll_bq(lens_run_id, args.timeout)
    elapsed = time.monotonic() - start

    # 3. Report
    print()
    if result is None:
        print(f"TIMEOUT — no BQ result after {elapsed:.1f}s.")
        print(f"  Check that the DGX backend is running and subscribed to: {TOPIC_ID}")
        sys.exit(2)

    print(f"Result received in {elapsed:.1f}s:")
    print(f"  run_id     : {result.get('run_id')}")
    print(f"  status     : {result.get('status')}")
    print(f"  latency_ms : {result.get('latency_ms')}")
    print(f"  error      : {result.get('error')}")
    print(f"  raw_row    : {str(result.get('raw_row', ''))[:200]}")
    fields_str = result.get("fields")
    if fields_str:
        try:
            parsed = json.loads(fields_str)
            print(f"  fields     : {json.dumps(parsed, indent=4)}")
        except json.JSONDecodeError:
            print(f"  fields     : {fields_str[:300]}")
    print("\nPASS")


if __name__ == "__main__":
    main()

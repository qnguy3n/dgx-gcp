"""Publishes a small batch of test messages to the configured Pub/Sub topic,
for exercising the pipeline end to end.

Uses the same topic/project as the running service (via config.load_config),
but published with *your own* gcloud user credentials, not the service
account in credentials/ -- that account only has subscribe permissions, not
publish. Run `gcloud auth login` first.

Payload shape matches what a real producer sends:
    {
        "run_id": "...",
        "project_id": "...",       -- which clinical project/study this letter belongs to
        "letter_id": "...",        -- id of the source letter within that project
        "datetime_sent": "...",    -- ISO-8601 UTC, when this message was published
        "letter_body": "...",
    }
`datetime_sent` is what the backend uses to measure transit latency (publish
-> arrival on this machine); see consume.py.

Usage:
    LENS_DGX_ENV=dev python tools/publish_test_message.py [--count N] [--project-id ID]
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone

# Make the parent dir's config.py importable when run as `python tools/publish_test_message.py`.
sys.path.insert(0, __file__.rsplit("/", 2)[0])

from google.cloud import pubsub_v1

from config import load_config

_SAMPLE_BODIES = [
    "Pt reports blurred vision for 3 weeks. VA (sc): OD 6/9, OS 6/12. No pain or redness noted.",
    "Routine diabetic eye check. Visual acuity with correction: right eye 20/20, left eye 20/25. Fundus exam unremarkable.",
    "Follow-up post-cataract surgery, left eye. VA (cc) OS 20/30, improving. No complications reported.",
]


def _build_payload(index: int, project_id: str) -> dict:
    return {
        "run_id": uuid.uuid4().hex,
        "project_id": project_id,
        "letter_id": f"LETTER-{index:04d}-{uuid.uuid4().hex[:8]}",
        "datetime_sent": datetime.now(timezone.utc).isoformat(),
        "letter_body": _SAMPLE_BODIES[index % len(_SAMPLE_BODIES)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=3, help="number of messages to publish (default: 3)")
    parser.add_argument("--project-id", default="demo-clinic-001", help="project_id field to stamp on each message")
    args = parser.parse_args()

    config = load_config()
    publisher = pubsub_v1.PublisherClient()
    # The topic and its subscription happen to share the same id on this
    # project (see config.py's subscription_id default).
    topic_path = publisher.topic_path(config.project_id, config.subscription_id)

    futures = []
    payloads = []
    for i in range(args.count):
        payload = _build_payload(i, args.project_id)
        payloads.append(payload)
        data = json.dumps(payload).encode("utf-8")
        futures.append(publisher.publish(topic_path, data))

    for payload, future in zip(payloads, futures):
        message_id = future.result(timeout=10)
        print(
            f"published run_id={payload['run_id']} letter_id={payload['letter_id']} "
            f"message_id={message_id} datetime_sent={payload['datetime_sent']} topic={topic_path}"
        )


if __name__ == "__main__":
    main()

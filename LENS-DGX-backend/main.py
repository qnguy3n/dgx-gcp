"""Entry point: subscribes to Pub/Sub and processes messages until killed.

Usage:
    LENS_DGX_ENV=dev python main.py

Run create_table.py once per environment first to make sure the BigQuery
table exists.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor

from google.cloud import bigquery, pubsub_v1
from google.cloud.pubsub_v1.subscriber.scheduler import ThreadScheduler
from google.cloud.pubsub_v1.types import FlowControl

from config import load_config
from consume import make_callback
from test_logger import InferenceLogger

logging.basicConfig(
    level=os.environ.get("LENS_DGX_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("lens_dgx.main")


def main() -> int:
    config = load_config()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.credentials_path
    logger.info(
        "starting lens-dgx-backend service: env=%s project=%s subscription=%s table=%s vllm=%s concurrency=%d",
        config.env, config.project_id, config.subscription_id, config.bq_table_id,
        config.vllm_url, config.max_concurrency,
    )

    bq_client = bigquery.Client()
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(config.project_id, config.subscription_id)

    inference_log = InferenceLogger(config.log_dir)
    callback = make_callback(config, bq_client, inference_log)

    executor = ThreadPoolExecutor(max_workers=config.max_concurrency)
    scheduler = ThreadScheduler(executor=executor)
    flow_control = FlowControl(max_messages=config.max_concurrency)

    streaming_pull_future = subscriber.subscribe(
        subscription_path, callback=callback, flow_control=flow_control, scheduler=scheduler,
    )
    logger.info("listening on %s (log: %s) ...", subscription_path,
                os.path.join(config.log_dir, "inference_log.jsonl"))

    def _shutdown(signum, frame):
        logger.info("received signal %s, shutting down...", signum)
        streaming_pull_future.cancel()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    with subscriber:
        try:
            streaming_pull_future.result()
        except Exception as exc:  # noqa: BLE001 - stream ended, log and exit cleanly
            logger.info("stream ended: %s", exc)

    logger.info("shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

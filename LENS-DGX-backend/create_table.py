"""Creates the codcot results table if it doesn't already exist.

Run this once per environment before starting the service:
    CODCOT_ENV=dev python create_table.py

Schema is deliberately loose: only run_id is guaranteed to exist on every
row, since the model's CSV output columns vary per run. See write_bq.py
for the full column rationale.
"""
from __future__ import annotations

import os

from google.cloud import bigquery

from config import load_config


def main() -> None:
    config = load_config()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.credentials_path

    client = bigquery.Client()
    schema = [
        bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("row_index", "INTEGER"),
        bigquery.SchemaField("fields", "JSON"),
        bigquery.SchemaField("raw_row", "STRING"),
        bigquery.SchemaField("status", "STRING"),
        bigquery.SchemaField("error", "STRING"),
        bigquery.SchemaField("model_name", "STRING"),
        bigquery.SchemaField("latency_ms", "FLOAT"),
        bigquery.SchemaField("processed_at", "TIMESTAMP"),
    ]
    table = bigquery.Table(config.bq_table_id, schema=schema)
    table = client.create_table(table, exists_ok=True)
    print(f"Table ready: {table.project}.{table.dataset_id}.{table.table_id}")


if __name__ == "__main__":
    main()

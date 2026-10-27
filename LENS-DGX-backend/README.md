# LENS-DGX-backend

Long-running pipeline: Pub/Sub → vLLM (`openai/gpt-oss-20b`) → BigQuery.

Lives as a sibling of `../codcot_tests/`, which is where this started as a
set of one-off test scripts (`pull.py`, `push.py`, `create_bq_table.py`) —
this folder holds the real service built on top of them. It reuses
`codcot_tests/.venv` and `codcot_tests/{dev,prod}_auth.json` rather than
duplicating them.

Each Pub/Sub message is `{"run_id": ..., "letter_body": "<instruction + text>"}`
(same shape as `../codcot_tests/push.py`). The instruction is already embedded in the
body, so it's sent to vLLM verbatim as the user turn — no prompt is added
here. The model's response is expected to look like a CSV (header row +
data rows); columns aren't fixed ahead of time, so they're stored as a
BigQuery `JSON` column rather than named columns.

## Files

| File | Role |
|---|---|
| `config.py` | Env-var driven config (dev/prod project, subscription, table, vLLM URL) |
| `main.py` | Entry point — opens the Pub/Sub streaming pull and blocks |
| `consume.py` | Per-message callback: parse → infer → parse CSV → write BQ → ack/nack |
| `infer.py` | HTTP call to the vLLM OpenAI-compatible endpoint |
| `csv_parse.py` | Parses the model's CSV-like output into row dicts |
| `write_bq.py` | Builds BigQuery rows and does the streaming insert |
| `create_table.py` | One-time table creation/migration |
| `test_logger.py` | Writes `logs/inference_log.jsonl` — one line per message with input, output, and timing, for testing |

## BigQuery schema

Only `run_id` is guaranteed to exist on every row (per the model's output
being free-form CSV):

```
run_id       STRING    REQUIRED
row_index    INTEGER             -- position within this message's CSV output
fields       JSON                -- parsed CSV row as key/value, columns vary by run
raw_row      STRING              -- original CSV line (full raw output if parsing failed)
status       STRING              -- ok | parse_error | inference_error | bad_message
error        STRING
model_name   STRING
latency_ms   FLOAT               -- vLLM inference time for this message
processed_at TIMESTAMP
```

Query example for a specific run: `SELECT run_id, JSON_VALUE(fields, '$.category') FROM ... WHERE run_id = '...'`.

## Setup

```bash
cd LENS-DGX-backend
source ../codcot_tests/.venv/bin/activate     # already has google-cloud-pubsub/bigquery + requests
CODCOT_ENV=dev python create_table.py         # one-time, per environment
```

## Run

```bash
CODCOT_ENV=dev python main.py
```

Env vars (all optional in dev — they default to the values already used by
`../codcot_tests/pull.py` / `push.py` / `create_bq_table.py`; `prod` requires
`CODCOT_PROJECT_ID`, `CODCOT_SUBSCRIPTION_ID`, `CODCOT_BQ_TABLE_ID` explicitly
since no confirmed prod values exist yet):

| Var | Default (dev) | Meaning |
|---|---|---|
| `CODCOT_ENV` | `dev` | `dev` or `prod` |
| `CODCOT_PROJECT_ID` | `ml-model-runner-f22694eb` | GCP project |
| `CODCOT_SUBSCRIPTION_ID` | `lens-dgx-spark-dev-input` | Pub/Sub subscription |
| `GOOGLE_APPLICATION_CREDENTIALS` | `../codcot_tests/dev_auth.json` | Service account key |
| `CODCOT_BQ_TABLE_ID` | `ml-model-runner-f22694eb.lens_dgx_spark_dev.codcot_results` | Output table |
| `CODCOT_VLLM_URL` | `http://localhost:8000` | vLLM server |
| `CODCOT_MODEL_NAME` | `openai/gpt-oss-20b` | Must match `--served-model-name` / model tag vLLM was started with |
| `CODCOT_MAX_CONCURRENCY` | `4` | Max in-flight messages / vLLM requests at once |
| `CODCOT_REQUEST_TIMEOUT_S` | `120` | Per-request timeout to vLLM |
| `CODCOT_MAX_OUTPUT_TOKENS` | `2048` | `max_tokens` sent to vLLM |
| `CODCOT_LOG_DIR` | `LENS-DGX-backend/logs` | Where `inference_log.jsonl` is written |
| `CODCOT_LOG_LEVEL` | `INFO` | Python logging level |

## Testing end-to-end

1. Start the vLLM server (`../setup_vllm_openai.txt`).
2. `CODCOT_ENV=dev python create_table.py`
3. `CODCOT_ENV=dev python main.py` (leave running in one terminal)
4. In another shell, publish a test message — note `push.py` needs your own
   `gcloud auth login`, since the service-account keys here can't publish:
   ```bash
   python ../codcot_tests/push.py
   ```
5. Watch:
   ```bash
   tail -f logs/inference_log.jsonl | python3 -m json.tool --compact
   ```
   Each line has `run_id`, `status`, `input_preview`, `output_preview`,
   `inference_latency_ms` (vLLM call only) and `total_processing_ms`
   (full message handling, receipt to ack).
6. Confirm the row(s) landed in BigQuery:
   ```sql
   SELECT * FROM `ml-model-runner-f22694eb.lens_dgx_spark_dev.codcot_results`
   ORDER BY processed_at DESC LIMIT 10;
   ```

## Deploying as a systemd service

See `codcot-service.example.service` in this directory — copy it to
`/etc/systemd/system/codcot-service.service`, adjust `User`/paths, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now codcot-service
journalctl -u codcot-service -f
```

## Known gaps / follow-ups

- `CODCOT_MODEL_NAME` must match whatever model tag vLLM was actually
  started with (`openai/gpt-oss-20b` per `setup_vllm_openai.txt`).
- The Pub/Sub subscription's dead-letter policy / max-delivery-attempts
  hasn't been checked — worth confirming so a permanently-failing message
  can't loop forever on the `nack()` path (transient errors only).
- No prod subscription/table id is known yet; `CODCOT_ENV=prod` requires
  them to be supplied explicitly (see table above).

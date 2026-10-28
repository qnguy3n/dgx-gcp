# LENS-DGX-backend

Long-running pipeline: Pub/Sub → vLLM (`openai/gpt-oss-20b`) → BigQuery.

Fully self-contained: its own `.venv`, its own copies of the service-account
keys (`credentials/`), its own test-publish tool (`tools/`). Nothing here
reaches outside this folder.

## Architecture: LENS ↔ DGX-GCP

This service is the **DGX-side worker** in a two-machine setup.  The LENS
web application (running on a separate machine with no direct network path to
the DGX GPU server) uses GCP as the communication channel:

```
LENS machine                          DGX machine
────────────────                      ────────────────────────────
LENS backend                          LENS-DGX-backend (this repo)
  │                                     │
  │  Pub/Sub publish (one msg/letter)   │
  ├──────────────────────────────────►  │  subscribe → vLLM → BigQuery
  │                                     │
  │  BigQuery poll (STARTS_WITH run_id))│
  ◄─────────────────────────────────────┤
```

**Message run_id convention**: LENS sets `run_id = "{lens_run_id}:{letter_id}"`.
This lets LENS query all rows for a run with `STARTS_WITH(run_id, "{lens_run_id}:")`
and recover the individual letter ID from the suffix — no BigQuery schema change
needed on this side.

Each Pub/Sub message is:

```json
{
    "run_id": "{lens_run_id}:{letter_id}",
    "project_id": "...",      // LENS study_id
    "letter_id": "...",       // row ID column value from LENS
    "datetime_sent": "...",   // ISO-8601 UTC, set by LENS at publish time
    "letter_body": "<instruction + text>"  // fully-formatted prompt
}
```

`project_id`, `letter_id` and `datetime_sent` are optional (older-shaped
messages without them still work) but are threaded through to
`logs/inference_log.jsonl` when present — `datetime_sent` in particular is
used to measure transit latency (see Timing below). The instruction is
already embedded in `letter_body`, so it's sent to vLLM verbatim as the user
turn — no prompt is added here.

The model's response can be **JSON** (preferred, as LENS prompts produce) or
**CSV** (header row + data rows). `csv_parse.py` tries JSON first and falls
back to CSV, so both shapes are stored correctly in BigQuery.

## Timing

Every message gets a four-stage timing breakdown, logged to both the regular
application log and `logs/inference_log.jsonl`:

| Field | Measures |
|---|---|
| `transit_latency_ms` | Producer's `datetime_sent` → our own wall-clock `received_at` (Pub/Sub delivery time, not our own processing) |
| `inference_latency_ms` | The vLLM call only |
| `bq_insert_latency_ms` | The BigQuery streaming-insert call only |
| `total_processing_ms` | Full handling: receipt to ack/nack |

`transit_latency_ms` is only computed when the message carries a
`datetime_sent` (see `consume.py::_transit_latency_ms`); it depends on this
machine's clock being roughly in sync with the producer's.

## Files

| File | Role |
|---|---|
| `config.py` | Env-var driven config (dev/prod project, subscription, table, vLLM URL) |
| `main.py` | Entry point — opens the Pub/Sub streaming pull and blocks |
| `consume.py` | Per-message callback: parse → infer → parse output → write BQ → ack/nack |
| `infer.py` | HTTP call to the vLLM OpenAI-compatible endpoint |
| `csv_parse.py` | Parses model output: JSON first, then CSV fallback |
| `write_bq.py` | Builds BigQuery rows and does the streaming insert |
| `create_table.py` | One-time table creation/migration |
| `test_logger.py` | Writes `logs/inference_log.jsonl` — one line per message with input, output, and timing, for testing |
| `tools/publish_test_message.py` | Publishes a batch of test messages (default 3), for exercising the pipeline end to end |
| `credentials/` | Service-account keys (`dev_auth.json`, `prod_auth.json`) — not committed, see `.gitignore` |

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
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
LENS_DGX_ENV=dev python create_table.py       # one-time, per environment
```

## Run

```bash
source .venv/bin/activate
LENS_DGX_ENV=dev python main.py
```

Env vars (all optional in dev — they default to this project's known dev
values; `prod` requires `LENS_DGX_PROJECT_ID`, `LENS_DGX_SUBSCRIPTION_ID`,
`LENS_DGX_BQ_TABLE_ID` explicitly since no confirmed prod values exist yet):

| Var | Default (dev) | Meaning |
|---|---|---|
| `LENS_DGX_ENV` | `dev` | `dev` or `prod` |
| `LENS_DGX_PROJECT_ID` | `ml-model-runner-f22694eb` | GCP project |
| `LENS_DGX_SUBSCRIPTION_ID` | `lens-dgx-spark-dev-input` | Pub/Sub subscription |
| `GOOGLE_APPLICATION_CREDENTIALS` | `credentials/dev_auth.json` | Service account key |
| `LENS_DGX_BQ_TABLE_ID` | `ml-model-runner-f22694eb.lens_dgx_spark_dev.results` | Output table |
| `LENS_DGX_VLLM_URL` | `http://localhost:8000` | vLLM server |
| `LENS_DGX_MODEL_NAME` | `openai/gpt-oss-20b` | Must match `--served-model-name` / model tag vLLM was started with |
| `LENS_DGX_MAX_CONCURRENCY` | `4` | Max in-flight messages / vLLM requests at once |
| `LENS_DGX_REQUEST_TIMEOUT_S` | `120` | Per-request timeout to vLLM |
| `LENS_DGX_MAX_OUTPUT_TOKENS` | `2048` | `max_tokens` sent to vLLM |
| `LENS_DGX_LOG_DIR` | `LENS-DGX-backend/logs` | Where `inference_log.jsonl` is written |
| `LENS_DGX_LOG_LEVEL` | `INFO` | Python logging level |

## Testing end-to-end

1. Start the vLLM server.
2. `LENS_DGX_ENV=dev python create_table.py`
3. `LENS_DGX_ENV=dev python main.py` (leave running in one terminal)
4. In another shell, publish a batch of test messages (default 3, override
   with `--count`) — note this needs your own `gcloud auth login`, since the
   service-account keys here can't publish:
   ```bash
   LENS_DGX_ENV=dev python tools/publish_test_message.py
   LENS_DGX_ENV=dev python tools/publish_test_message.py --count 2 --project-id my-study-001
   ```
5. Watch:
   ```bash
   tail -f logs/inference_log.jsonl | python3 -m json.tool --compact
   ```
   Each line has `run_id`, `project_id`, `letter_id`, `status`,
   `input_preview`, `output_preview`, and the full timing breakdown
   (`transit_latency_ms`, `inference_latency_ms`, `bq_insert_latency_ms`,
   `total_processing_ms` — see Timing above).
6. Confirm the row(s) landed in BigQuery:
   ```sql
   SELECT * FROM `ml-model-runner-f22694eb.lens_dgx_spark_dev.results`
   ORDER BY processed_at DESC LIMIT 10;
   ```

## Deploying as a systemd service

See `lens-dgx-backend.example.service` in this directory — copy it to
`/etc/systemd/system/lens-dgx-backend.service`, adjust `User`/paths, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now lens-dgx-backend
journalctl -u lens-dgx-backend -f
```

## Known gaps / follow-ups

- `LENS_DGX_MODEL_NAME` must match whatever model tag vLLM was actually
  started with (`openai/gpt-oss-20b`).
- The Pub/Sub subscription's dead-letter policy / max-delivery-attempts
  hasn't been checked — worth confirming so a permanently-failing message
  can't loop forever on the `nack()` path (transient errors only).
- No prod subscription/table id is known yet; `LENS_DGX_ENV=prod` requires
  them to be supplied explicitly (see table above).

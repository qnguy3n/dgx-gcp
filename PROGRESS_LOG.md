# Progress Log

Chronological record of work on this repo and the DGX Spark deployment it
documents. Latest entries on top.

## 2026-10-27 — Realistic payload + timing instrumentation, BigQuery insert bug fix

- `LENS-DGX-backend/tools/publish_test_message.py`: payload now carries
  `project_id`, `letter_id`, and `datetime_sent` alongside `run_id`/
  `letter_body`, matching a real producer's message shape. Publishes a
  batch of messages (default 3, `--count` to override) instead of one.
- `LENS-DGX-backend/consume.py`: added a full per-message timing
  breakdown, logged to the app log and `logs/inference_log.jsonl`:
  - `transit_latency_ms` — producer's `datetime_sent` → this machine's
    wall-clock `received_at` (Pub/Sub delivery time)
  - `inference_latency_ms` — the vLLM call only (already existed)
  - `bq_insert_latency_ms` — the BigQuery insert call, newly timed at
    every insert site
  - `total_processing_ms` — full handling, receipt to ack/nack (already
    existed)
  - `project_id`/`letter_id`/`datetime_sent` are threaded through when
    present; older-shaped messages without them still work.
- `LENS-DGX-backend/test_logger.py`: `InferenceLogger.log()` extended to
  accept and record the fields above.
- **Bug found and fixed**: `write_bq.py` passed the parsed CSV row as a
  raw Python dict into BigQuery's `fields` column (type `JSON`) via the
  legacy streaming-insert API, which rejects nested dicts for JSON-typed
  columns (`"This field: fields is not a record"`). Every message had been
  failing to insert and nacking for retry; the results table had sat at
  0 rows this entire deployment because of it. Fixed by `json.dumps()`-
  encoding the value before insert.
- Verified for real: ran `consume.py`'s actual callback (not a mock)
  against the live vLLM server and the live BigQuery table with 3 fake
  messages — confirmed all 3 acked, timing fields correct (including a
  deliberately simulated ~150ms transit delay), and queried BigQuery
  directly to confirm the rows landed with correctly-structured `fields`.
  (Still untested: the actual Pub/Sub transport end-to-end — no
  `gcloud auth login` is set up on this machine to publish for real, so
  only the transport layer itself is unverified.)
- `lens-dgx-backend` systemd service restarted to pick up all of the
  above.

## 2026-10-27 — Fixed stale systemd path after LENS-DGX-backend moved into dgx-gcp/

- `LENS-DGX-backend/` (and `gpu_test/`) were physically relocated from
  the top-level `LENS/` directory into `LENS/dgx-gcp/` to consolidate the
  working tree under this repo. The running backend process was
  unaffected (same inode, kernel-tracked rename), but the installed
  systemd unit still had the old path baked into `WorkingDirectory` and
  `ExecStart`, which would have failed on the next restart/reboot.
- Updated `lens-dgx-backend.example.service` to the new
  `dgx-gcp/LENS-DGX-backend` path, re-installed it, and restarted the
  service to confirm it comes up cleanly from the new location.

## 2026-10-27 — Made LENS-DGX-backend self-contained; renamed codcot* → LENS_DGX_*

- Backend previously reached outside its own folder for its Python venv
  and GCP service-account keys (`../codcot_tests/`). Made it fully
  self-contained:
  - Own `.venv/` (`requirements.txt`: `google-cloud-pubsub`,
    `google-cloud-bigquery`, `requests`)
  - Own `credentials/` holding `dev_auth.json`/`prod_auth.json`
    (`chmod 600`, excluded from git via `.gitignore`)
  - `tools/publish_test_message.py` replaces the old
    `../codcot_tests/push.py` dependency
- Renamed the inherited `CODCOT_*` naming (traced to the original
  one-off scripts at `/home/moorfields1/codcot_tests/`, never actually
  defined anywhere as an acronym) to `LENS_DGX_*` throughout: env vars,
  logger namespaces (`lens_dgx.main`/`lens_dgx.consume`), log messages,
  docstrings.
- Renamed the BigQuery table `codcot_results` → `results` (dataset is
  already `lens_dgx_spark_dev`, so the qualified name reads cleanly).
  Confirmed the old table was empty (0 rows) before dropping it and
  creating the renamed one.
- `codcot-service.example.service` → `lens-dgx-backend.example.service`,
  `ExecStart` pointed at the new local `.venv`.
- Installed as `lens-dgx-backend.service` via systemd (`enable --now`),
  replacing the ad hoc `nohup`'d process from the session before.

## 2026-10-27 — Fixed the repo's broken first commit; pushed to GitHub

- Found and corrected two problems in `dgx-gcp`'s original `first commit`
  (`1608af0`, since rewritten as `d4631f2`):
  - `gpu_test/` had been committed as an empty git submodule reference
    (mode `160000`) instead of real files — a leftover nested `.git` at
    `git add` time made git record it as a gitlink pointing to a commit
    that doesn't exist in this repo's history. A fresh clone would have
    gotten an empty `gpu_test/` directory.
  - `.gitignore` existed on disk but hadn't been staged for the original
    commit, so `LENS-DGX-backend/__pycache__/*.pyc` got committed.
  - Also excluded `service.out` (a transient `nohup` log, not project
    content).
- Reset the commit (keeping the working tree), re-staged, and committed
  again with no content changes beyond the above — root commit, same
  single-commit history, force-pushed to `github.com:qnguy3n/dgx-gcp.git`.

## 2026-10-27 — Brought up vLLM + LENS-DGX-backend, set up boot persistence

- Started the `gpt-oss-vllm` Docker container
  (`vllm/vllm-openai:latest`, `openai/gpt-oss-20b`, `mxfp4` quantization)
  on `:8000`; confirmed healthy via `/health` and `/v1/models`.
- Worked around a missing `/tmp/encodings` mount (the documented
  tiktoken-vocab fix in `setup_vllm_openai.txt` assumes it exists) by
  fetching the encoding files fresh and, for reboot-safety, moving them
  to a persistent `~/.cache/vllm-tiktoken-encodings` instead of `/tmp`
  (which is subject to a 30-day cleanup policy on this machine).
- Ran `create_table.py` (dev) and started `LENS-DGX-backend/main.py`,
  confirmed it holds a live streaming-pull connection to the Pub/Sub
  subscription (not polling).
- Set up reboot persistence for both: recreated the vLLM container with
  `--restart unless-stopped` (incompatible with the original `--rm`), and
  installed the backend as a systemd service
  (`codcot-service.service` at the time, later renamed — see above).

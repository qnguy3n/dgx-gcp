# dgx-gcp

Code and tooling for running LENS inference on a DGX GPU server via GCP,
where the LENS VM and the DGX machine have no direct network path to each other.

## Repository layout

```
LENS-DGX-backend/   Long-running worker: Pub/Sub → vLLM → BigQuery (see its own README)
gpu_test/           GPU inference benchmarks: vLLM vs Ollama on the DGX Spark (GB10)
docs/               Latency analysis reports (batch size vs latency, roundtrip timing)
examples/           Original one-off Pub/Sub + BigQuery test scripts (pre-service)
PROGRESS_LOG.md     Chronological record of work and deployment milestones
```

## Architecture

```
LENS VM                               DGX machine
────────────────────                  ────────────────────────────────────────
LENS backend (core/dgx.py)            LENS-DGX-backend/main.py
  │                                     │
  │  Pub/Sub publish (one msg/letter)   │
  ├──────────────────────────────────►  │  subscribe → vLLM → BigQuery insert
  │                                     │
  │  BigQuery poll (STARTS_WITH)        │
  ◄─────────────────────────────────────┤
```

The LENS VM publishes one Pub/Sub message per letter with
`run_id = "{lens_run_id}:{letter_id}"`. LENS polls BigQuery using
`STARTS_WITH(run_id, "{lens_run_id}:")` to retrieve all results for a run
without any schema change on the DGX side.

## Quick start

See `LENS-DGX-backend/README.md` for full setup, env vars, systemd deployment,
and end-to-end testing instructions.

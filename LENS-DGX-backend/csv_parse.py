"""Parses the model's output into rows.

Tries JSON first (a single dict or a list of dicts), then falls back to
CSV with a header row.  This makes the pipeline compatible with LENS
prompts that instruct the model to return JSON as well as any prompts that
ask for CSV.

The model's output columns aren't fixed -- only `run_id` is guaranteed to
exist on the BigQuery side -- so this returns a list of plain dicts keyed
by whatever the model produced, rather than assuming any particular columns.
"""
from __future__ import annotations

import csv
import io
import json
from typing import Dict, List


def parse_csv_output(raw_text: str) -> List[Dict[str, str]]:
    """Parse `raw_text` as JSON (preferred) or CSV with a header row.

    JSON path: accepts a single ``{...}`` dict (wrapped into a one-element
    list) or a ``[{...}, ...]`` list of dicts.  All values are coerced to
    strings to keep downstream handling uniform with the CSV path.

    CSV path: requires a header row and at least one non-empty data row.

    Raises ValueError if neither format produces usable rows, so the caller
    can fall back to storing the raw text with a ``parse_error`` status.
    """
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("empty model output")

    # --- JSON path ---
    # Strip optional markdown code fences (```json ... ``` or ``` ... ```)
    json_candidate = cleaned
    if json_candidate.startswith("```"):
        lines = json_candidate.splitlines()
        # drop first and last fence lines
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        json_candidate = "\n".join(inner).strip()

    if json_candidate.startswith("{") or json_candidate.startswith("["):
        try:
            parsed = json.loads(json_candidate)
            if isinstance(parsed, dict):
                return [{k: str(v) for k, v in parsed.items()}]
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return [{k: str(v) for k, v in row.items()} for row in parsed]
        except json.JSONDecodeError:
            pass  # fall through to CSV

    # --- CSV path ---
    reader = csv.DictReader(io.StringIO(cleaned))
    if not reader.fieldnames or all((f is None or not f.strip()) for f in reader.fieldnames):
        raise ValueError("no header row detected (not valid JSON or CSV)")

    rows = [row for row in reader if any((v or "").strip() for v in row.values())]
    if not rows:
        raise ValueError("no data rows after header")

    return rows

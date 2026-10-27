"""Parses the model's CSV-like output into rows.

The model's output columns aren't fixed -- only `run_id` is guaranteed to
exist on the BigQuery side -- so this returns a list of plain dicts keyed
by whatever header row the model produced, rather than assuming any
particular columns.
"""
from __future__ import annotations

import csv
import io
from typing import Dict, List


def parse_csv_output(raw_text: str) -> List[Dict[str, str]]:
    """Parse `raw_text` as CSV with a header row.

    Raises ValueError if the text doesn't look like usable CSV (no header,
    or no data rows) so the caller can fall back to storing the raw text
    with a parse_error status instead of silently losing it.
    """
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("empty model output")

    reader = csv.DictReader(io.StringIO(cleaned))
    if not reader.fieldnames or all((f is None or not f.strip()) for f in reader.fieldnames):
        raise ValueError("no header row detected")

    rows = [row for row in reader if any((v or "").strip() for v in row.values())]
    if not rows:
        raise ValueError("no data rows after header")

    return rows

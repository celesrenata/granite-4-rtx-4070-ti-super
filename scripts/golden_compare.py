#!/usr/bin/env python
"""Phase-1 golden compare CLI (stdlib only, no network, no downloads).

Thin wrapper around :mod:`reference.granite.golden` ``compare``.

    python scripts/golden_compare.py \
        --reference golden/reference.json \
        --candidate golden/engine.json --atol 1e-3 --rtol 1e-2 \
        [--report-out report.json]

Tolerances (``--atol`` / ``--rtol``) are caller-supplied and apply **only**
to the adapter-supplied numeric intermediate ``values``; token IDs and
tensor checksums are always compared exactly. Absent evidence (missing
records, intermediates, fields, or a checkpoint hash mismatch) yields a
``refused`` verdict, never a match.

Exit codes: 0 = match, 1 = divergent, 2 = refused / unusable input.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.granite.golden import main_compare  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main_compare())

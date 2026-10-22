#!/usr/bin/env python
"""Phase-1 golden capture CLI (stdlib only, no network, no downloads).

Thin wrapper around :mod:`reference.granite.golden` ``capture``.

    python scripts/golden_capture.py \
        --adapter tests/fake_granite_adapter.py \
        --checkpoint-dir /local/granite-4.2-8b \
        --prompts prompts.json --out golden/reference.json

Prerequisites are verified before anything runs:
  * ``--adapter``: a local ``.py`` file or importable dotted module that
    supplies ``ADAPTER_NAME``, ``SUPPORTED_SAMPLING`` and
    ``create_reference_adapter(checkpoint_dir)``;
  * ``--checkpoint-dir``: a local directory with readable JSON
    ``config.json`` and ``tokenizer_config.json``;
  * ``--prompts``: a JSON array of prompt objects.

Unsupported sampling modes and absent prerequisites are refused (exit
code 2); nothing is written on refusal.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.granite.golden import main_capture  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main_capture())

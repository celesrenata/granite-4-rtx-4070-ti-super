# Phase-1 golden reference

**Status: tooling only — no real Golden outputs exist yet.**

The Phase-1 golden capture/compare tooling is implemented and hermetically
tested, but **no real Golden outputs have been measured**. Real captures
require two local prerequisites that are not present in this repository and
are **never downloaded by the tooling**:

1. A **local Granite reference adapter** module — a deterministic local
   implementation that loads the real checkpoint and exposes the golden
   adapter protocol (`ADAPTER_NAME`, `SUPPORTED_SAMPLING`,
   `create_reference_adapter(checkpoint_dir)`; see below);
2. A **local checkpoint directory** containing the real Granite 4.2 8B
   `config.json` and `tokenizer_config.json` (weights live inside the
   directory; they are never opened by the capture utility itself).

Until both are supplied, every real capture is *refused by design*: the
tool rejects absent prerequisites and unsupported sampling instead of
claiming results.

## Components

| File | Role |
|---|---|
| [`reference/granite/golden.py`](../reference/granite/golden.py) | Stdlib-only capture/compare core (no torch, no network). |
| [`scripts/golden_capture.py`](../scripts/golden_capture.py) | `capture` CLI wrapper. |
| [`scripts/golden_compare.py`](../scripts/golden_compare.py) | `compare` CLI wrapper. |
| [`tests/fake_granite_adapter.py`](../tests/fake_granite_adapter.py) | Synthetic deterministic fake adapter (test fixture only; never used with real checkpoints). |
| [`tests/test_golden_reference.py`](../tests/test_golden_reference.py) | Hermetic tests: temp files, no torch/CUDA/network. |

## What a capture record contains

One JSON record per prompt (`granite4.golden-record.v1`):

- `prompt_id`;
- exact `input_ids` / `output_ids` token ID lists;
- `intermediates`: for each selected intermediate tensor *supplied by the
  adapter* — `shape`, `sha256` checksum, optional `checksum_kind` and
  optional numeric `values` (the adapter decides which tensors to report);
- `adapter` / library metadata (`name`, `module`, plus
  `adapter_metadata()` output);
- `captured_at_utc` (UTC timestamp);
- `checkpoint` — SHA256 hashes of the local `config.json` and
  `tokenizer_config.json` (plus optional file hashes when present).

The capture document is `granite4.golden-capture.v1`; a comparison report is
`granite4.golden-compare-report.v1`.

## What the comparator guarantees

- Token IDs and tensor checksums are compared **exactly**;
- caller-supplied tolerances (`--atol` / `--rtol`) apply **only** to the
  adapter-supplied numeric `values`;
- missing records, intermediates, or fields — or a checkpoint hash mismatch
  between the two captures — make the verdict **`refused`**, never
  `match`.

Exit codes (capture and compare CLIs): `0` match, `1` divergent,
`2` refused / unusable input.

## Adapter protocol

A real local Granite reference adapter must be a module (`.py` file or
dotted import name, supplied by the caller) with:

```python
ADAPTER_NAME = "granite-4.2-8b-local"
SUPPORTED_SAMPLING = frozenset({"greedy"})  # only modes it actually implements

def create_reference_adapter(checkpoint_dir): ...  # must reject a missing
                                                    # local checkpoint itself
```

The instance must expose `run(prompt_id, text, input_ids, sampling) ->
{"input_ids", "output_ids", "intermediates"}` and `adapter_metadata() ->
dict`. Modes that cannot run deterministically (stochastic sampling) must be
*absent* from `SUPPORTED_SAMPLING`; requesting them makes the capture
refuse with a clear error and writes nothing.

## Usage

```console
python scripts/golden_capture.py \
    --adapter /local/path/granite_reference_adapter.py \
    --checkpoint-dir /local/granite-4.2-8b \
    --prompts prompts.json --out golden/reference.json

python scripts/golden_compare.py \
    --reference golden/reference.json \
    --candidate golden/engine.json --atol 1e-3 --rtol 1e-2
```

`prompts.json` is a JSON array of objects: `prompt_id` plus exactly one of
`text` or `input_ids`, optional per-prompt `sampling`.

## How to test now

```console
python -m pytest tests/test_golden_reference.py
```

The suite uses only the fake adapter and temporary directories.

## Limitations

- No real model output is produced or compared today; "match" in tests
  means *the fake adapter is deterministic*, not that the Granite engine
  matches a reference.
- Intermediate tensors are whatever the adapter chooses to supply; there is
  no fixed tensor list until a real adapter exists.
- The capture utility validates checkpoint *files* (presence, JSON validity)
  but does not re-verify planning architecture fields — that is the Phase-0
  contract tooling (`reference/granite/contract.py`,
  `scripts/inspect_checkpoint.py`).

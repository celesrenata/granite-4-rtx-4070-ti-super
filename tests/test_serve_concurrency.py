"""ServeProcess multiplexes concurrent requests over one engine by request id."""
from __future__ import annotations

import sys
import textwrap
import threading
from pathlib import Path

from frontend.engine_service import ServeProcess

# A fake engine: reports ready, collects two requests, answers them in REVERSE
# order (as a concurrent engine may), echoing each request's id.
_FAKE_REORDERING = textwrap.dedent("""
    import json, sys
    print(json.dumps({"ready": True}), flush=True)
    reqs = [json.loads(sys.stdin.readline()) for _ in range(2)]
    for r in reversed(reqs):
        print(json.dumps({"id": r["id"], "generated": [r["tokens"][0] * 10]}), flush=True)
""")

# A pre-id engine: one request at a time, no id in the reply.
_FAKE_LEGACY = textwrap.dedent("""
    import json, sys
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        r = json.loads(line)
        print(json.dumps({"generated": [r["tokens"][0] + 1]}), flush=True)
""")


def _engine(tmp_path: Path, src: str) -> list[str]:
    script = tmp_path / "fake_engine.py"
    script.write_text(src)
    return [sys.executable, str(script)]


def test_concurrent_requests_get_their_own_replies(tmp_path: Path) -> None:
    sp = ServeProcess(_engine(tmp_path, _FAKE_REORDERING), startup_timeout_s=30)
    sp.warm()
    results: dict[int, dict] = {}

    def ask(tok: int) -> None:
        results[tok] = sp.request({"tokens": [tok], "max_tokens": 1}, timeout_s=30)

    threads = [threading.Thread(target=ask, args=(t,)) for t in (3, 7)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    sp.close()
    assert results[3] == {"generated": [30]}
    assert results[7] == {"generated": [70]}


def test_reply_without_id_goes_to_the_only_request(tmp_path: Path) -> None:
    sp = ServeProcess(_engine(tmp_path, _FAKE_LEGACY), startup_timeout_s=30)
    assert sp.request({"tokens": [4], "max_tokens": 1}, timeout_s=30) == {"generated": [5]}
    assert sp.request({"tokens": [9], "max_tokens": 1}, timeout_s=30) == {"generated": [10]}
    sp.close()

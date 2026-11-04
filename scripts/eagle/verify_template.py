#!/usr/bin/env python3
"""Verify gen_serving.render_prompt matches prep_selfgen.py's tokenization.

prep_selfgen.py is the ground truth (it already produced correct selfgen24).
We run it on a small test record, read the reference token IDs from tokens.bin,
then render the same record through gen_serving and compare.
"""
import json
import subprocess
import struct
import sys
from pathlib import Path

CKPT = "/opt/granite4-rtx-4070-ti-super/ckpt"
PREP = "/opt/granite4-rtx-4070-ti-super/eagle/prep_selfgen.py"
GEN = "/opt/granite4-rtx-4070-ti-super/eagle/gen_serving.py"
TMP = Path("/tmp/tmpltest")
if TMP.exists():
    for f in TMP.iterdir():
        f.unlink()
TMP.mkdir(parents=True)

PROMPT = "Write a 250-word essay about the sociology of open source"
REPLY = "Open source is a collaborative philosophy. It began in the software world, where developers share source code so anyone can study, modify, and redistribute it. This openness accelerates innovation, because problems are solved by many minds at once, and solutions are inspected and improved publicly. The sociological dimension is the community itself: norms of contribution, credit, and trust emerge without a central authority. Governance becomes a question of how to coordinate thousands of volunteers, resolve conflicts, and decide what belongs in the project. Incentives matter; reputation, employment, and ideology all drive participation. The result is a commons that is at once a technical artifact and a social institution, and its health depends as much on its social rules as on its code. Understanding open source therefore requires reading it as a practice of shared production, not merely as a licensing choice."

test = [{"prompt": PROMPT, "reply": REPLY, "finish": "stop"}]
(TMP / "data.json").write_text(json.dumps(test))

# 1) ground truth via prep_selfgen.py
subprocess.run([sys.executable, PREP, "--tokenizer", CKPT + "/tokenizer.json",
                "--data", str(TMP / "data.json"), "--out", str(TMP)],
               check=True, capture_output=True)
buf = (TMP / "tokens.bin").read_bytes()
n = int.from_bytes(buf[:4], "little")
ref = list(struct.unpack(f"<{n}i", buf[4:4 + 4 * n]))

# 2) gen_serving render
sys.path.insert(0, str(TMP.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("gen_serving", GEN)
gs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gs)
from tokenizers import Tokenizer
tok = Tokenizer.from_file(CKPT + "/tokenizer.json")
tmpl = gs.load_template(Path(PREP))
prompt = gs.render_prompt(tmpl, PROMPT)
pids = tok.encode(prompt, add_special_tokens=False).ids
rids = tok.encode(REPLY, add_special_tokens=False).ids
rids.append(100257)  # finish == stop
cand = pids + rids
# round-trip: engine token ids -> text -> ids must be identity (gen_serving
# decodes generated ids to text, prep_selfgen re-encodes them)
text = tok.decode(rids, skip_special_tokens=True)
rtok = tok.encode(text, add_special_tokens=False).ids + [100257]
roundtrip_ok = rtok == rids

print(f"ref len={len(ref)} cand len={len(cand)}")
if ref == cand:
    print("MATCH: rendered prompt + reply tokenizes identically to prep_selfgen.py")
else:
    print("MISMATCH")
    for i, (a, b) in enumerate(zip(ref, cand)):
        if a != b:
            print(f"  first diff at {i}: ref={a} cand={b}")
            print(f"  ref  ctx: {ref[max(0,i-3):i+4]}")
            print(f"  cand ctx: {cand[max(0,i-3):i+4]}")
            break
print("round-trip decode->re-encode identity:", roundtrip_ok)

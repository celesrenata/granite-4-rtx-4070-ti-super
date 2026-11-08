# Extract conversations (user turns) from UltraChat parquet + ShareGPT, for
# target self-generation. Output: [{"src", "turns": [user1, user2?]}]
import argparse, json, random, glob
ap = argparse.ArgumentParser()
# The shuffle is seeded, so disjoint [start, start + count) slices never overlap.
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--count", type=int, default=2400)
ap.add_argument("--out", default="/w/eagle/ds/chat_prompts.json")
args = ap.parse_args()
import pyarrow.parquet as pq
random.seed(20261107)
out = []
for f in sorted(glob.glob("/w/eagle/data/ultrachat_train_sft-*.parquet")):
    for msgs in pq.read_table(f, columns=["messages"]).column("messages").to_pylist():
        users = [m["content"] for m in msgs if m["role"] == "user"]
        if users and 20 <= len(users[0]) <= 6000:
            out.append({"src": "ultrachat", "turns": users[:2]})
sg = json.load(open("/w/eagle/data/sharegpt_v43.json"))
for c in sg:
    conv = c.get("conversations") or []
    users = [t.get("value", "") for t in conv if t.get("from") == "human"]
    if users and 20 <= len(users[0]) <= 6000 and users[0].isascii():
        out.append({"src": "sharegpt", "turns": users[:2]})
random.shuffle(out)
print(len(out), "candidate conversations", sum(o["src"] == "sharegpt" for o in out), "sharegpt")
# Keep 2 turns only where a usable second user turn exists, ~40% of the set.
sel = []
for o in out[args.start:args.start + args.count]:
    t = [o["turns"][0]]
    if len(o["turns"]) > 1 and 10 <= len(o["turns"][1]) <= 4000 and random.random() < 0.55:
        t.append(o["turns"][1])
    sel.append({"src": o["src"], "turns": t})
json.dump(sel, open(args.out, "w"))
print(len(sel), "selected;", sum(len(s["turns"]) == 2 for s in sel), "two-turn")

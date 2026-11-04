#!/usr/bin/env python3
"""Live end-to-end checks for the granite4-rtx service (stdlib only)."""
import json
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:2701"
MODEL = "granite-4.2-8b"
results = []


def call(method, path, body=None, timeout=300):
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, json.loads(raw) if raw else None, time.time() - t0
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw), time.time() - t0
        except json.JSONDecodeError:
            return e.code, raw.decode(errors="replace"), time.time() - t0


def chat(messages, **kw):
    return call("POST", "/v1/chat/completions", {"model": MODEL, "messages": messages, **kw})


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def text(body):
    return body["choices"][0]["message"]["content"]


def user(content):
    return [{"role": "user", "content": content}]


# --- basic endpoints -------------------------------------------------------
s, b, _ = call("GET", "/health")
check("health 200 + available", s == 200 and b["inference"] == "available", json.dumps(b))
s, b, _ = call("GET", "/v1/models")
check("models lists granite-4.2-8b", s == 200 and b["data"][0]["id"] == MODEL, json.dumps(b)[:120])
s, b, _ = call("GET", "/metrics")
check("metrics 200", s == 200)
s, b, _ = call("GET", "/nope")
check("unknown path 404", s == 404, str(s))

# --- answers -----------------------------------------------------------------
s, b, dt = chat(user("What is the capital of France? Answer in one sentence."))
check("factual answer", s == 200 and "Paris" in text(b), f"{text(b)!r} {dt:.2f}s")
check("finish_reason stop + no <|im_end|> in text",
      b["choices"][0]["finish_reason"] == "stop" and "<|im_end|>" not in text(b))
check("usage arithmetic", b["usage"]["total_tokens"] == b["usage"]["prompt_tokens"] + b["usage"]["completion_tokens"])

s, b, _ = chat(user("What is 17 * 23? Reply with just the number."))
check("arithmetic", s == 200 and "391" in text(b), repr(text(b)))

s, b, _ = chat([{"role": "system", "content": "You only ever reply in French."},
                {"role": "user", "content": "Say good morning."}])
check("system prompt honoured", s == 200 and any(w in text(b).lower() for w in ("bonjour", "matin")), repr(text(b)))

s, b, _ = chat([{"role": "user", "content": "My name is Ada. Remember it."},
                {"role": "assistant", "content": "Got it, Ada."},
                {"role": "user", "content": "What is my name? One word."}])
check("multi-turn context", s == 200 and "Ada" in text(b), repr(text(b)))

s, b, _ = chat(user("Write a Python function that returns the nth Fibonacci number."), max_tokens=300)
check("code generation", s == 200 and "def " in text(b), f"{b['usage']['completion_tokens']} tokens")

s, b, _ = chat(user("Reply with a JSON object with keys name and age for a 30 year old named Bob. JSON only."))
t = text(b).strip().strip("`").removeprefix("json").strip()
try:
    obj = json.loads(t)
    ok = obj.get("name") == "Bob" and obj.get("age") == 30
except Exception:
    ok = False
check("JSON output parses", ok, repr(text(b))[:120])

s, b, _ = chat(user("Say hello in Japanese and Russian."))
check("unicode output", s == 200 and any(ord(c) > 0x3000 for c in text(b)) and any(0x400 <= ord(c) < 0x500 for c in text(b)), repr(text(b)))

# --- determinism --------------------------------------------------------------
m = user("Describe the ocean in two sentences.")
a = chat(m)[1]
b2 = chat(m)[1]
check("greedy determinism (same request twice)", text(a) == text(b2))

# --- limits -------------------------------------------------------------------
s, b, _ = chat(user("Count from 1 to 100, comma separated."), max_tokens=7)
check("max_tokens honoured -> length", s == 200 and b["usage"]["completion_tokens"] == 7
      and b["choices"][0]["finish_reason"] == "length", json.dumps(b["usage"]))

s, b, dt = chat(user("Write a detailed essay about the history of computing."), max_tokens=1000)
n = b["usage"]["completion_tokens"]
check("long generation", s == 200 and n > 300, f"{n} tokens in {dt:.2f}s = {n / dt:.1f} tok/s")

long_text = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(500))
s, b, dt = chat(user(long_text + "\nHow many items were listed? Answer with the number only."), max_tokens=16)
check("long prompt (~7K tokens)", s == 200, f"prompt {b['usage']['prompt_tokens'] if s == 200 else '?'} tok, {dt:.2f}s, answer {text(b)!r}" if s == 200 else str(b))

s, b, _ = chat(user("word " * 9000))
check("over-long prompt -> 400 context_length_exceeded", s == 400 and b["error"]["code"] == "context_length_exceeded", str(b)[:150])

# --- validation -----------------------------------------------------------------
for name, body in [
    ("empty messages", {"model": MODEL, "messages": []}),
    ("bad role", {"model": MODEL, "messages": [{"role": "robot", "content": "x"}]}),
    ("max_tokens 0", {"model": MODEL, "messages": user("hi"), "max_tokens": 0}),
    ("max_tokens string", {"model": MODEL, "messages": user("hi"), "max_tokens": "5"}),
    ("missing messages", {"model": MODEL}),
]:
    s, b, _ = call("POST", "/v1/chat/completions", body)
    check(f"400 on {name}", s == 400, str(s))
s, b, _ = call("POST", "/v1/chat/completions", b"{not json")
check("400 on malformed JSON", s == 400, str(s))

# --- concurrency ----------------------------------------------------------------
out = {}


def worker(i):
    out[i] = chat(user(f"What is {i} + {i}? Reply with just the number."))


t0 = time.time()
ths = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
for t in ths:
    t.start()
for t in ths:
    t.join()
ok = all(out[i][0] == 200 and str(2 * i) in text(out[i][1]) for i in range(8))
check("8 concurrent requests all correct", ok,
      f"{time.time() - t0:.2f}s total; " + ", ".join(repr(text(out[i][1])) for i in range(8)))

print(f"\n{sum(o for _, o in results)}/{len(results)} passed")
sys.exit(0 if all(o for _, o in results) else 1)

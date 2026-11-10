# How a language model actually runs: a field guide through one real project

A companion to `journey.md`, written for engineers who build with AI every day (prompts, agents, RAG, API calls) but have never had to think about what happens on the GPU. It explains the ideas you need, using the numbers from this project: Granite 4.2 8B, served on one 16 GB RTX 4070 Ti SUPER, from a scaffold to a production service in about three weeks.

You don't need any math beyond multiplication. Each section ends with what it means for how you use models.

---

## 1. The model is a big pile of numbers, read once per token

An 8B model is 8 billion numbers ("weights"). Running it means taking your text, turning it into tokens (word pieces; about ¾ of a word each in English), and pushing each token through 40 identical layers. Each layer multiplies the token's current representation by its weight matrices.

The key fact is that **every new token the model writes requires reading all the weights once.** There's no shortcut. The model doesn't "know" the next word; it computes it by passing through every layer.

We store the weights in FP8 (one byte per number), so one token means reading **8.38 GB** of memory. The card reads memory at **632 GB/s**. That makes the speed limit simple arithmetic:

> 632 GB/s ÷ 8.38 GB per token ≈ **75 tokens per second, at best.**

The GPU can do far more math per second than this needs. It spends most of its time waiting for memory. Engineers call this being **memory-bandwidth-bound**. Our first working version did 18 tokens/s. After three weeks of fixes it does 72, which is 95% of that limit.

**What it means for you:** output tokens are slow and sequential, and each one is a full trip through the model. A 1,000-token answer takes about 1,000 of those trips. That's why "be concise" in a prompt is a speed setting, not just a style choice.

---

## 2. Reading the prompt is different from writing the answer

There are two phases in every request:

- **Prefill**: the model reads your whole prompt. All prompt tokens are known up front, so they go through the layers together, as one big batch. The weights are read once for the whole batch, so the work becomes math-heavy instead of memory-heavy, and the GPU's math units (the "tensor cores") get busy.
- **Decode**: the model writes the answer one token at a time, each needing the previous one. This is the memory-bound phase from section 1.

The numbers show the difference:

| | Speed |
| --- | ---: |
| Prefill, 1K-token prompt | ~3,300 tokens/s |
| Decode | ~68 tokens/s (plain) |

Reading is about 50 times faster per token than writing.

**What it means for you:** long inputs are cheap compared with long outputs, which is why API pricing usually charges more for output tokens. But very long inputs aren't free (see section 4).

---

## 3. The KV cache: the model's working memory

When the model writes token 501, it needs to "look back" at tokens 1–500. It does that through **attention**: each layer compares the new token against every earlier one and decides what's relevant.

Recomputing everything about the 500 earlier tokens at every step would be hopeless, so the model saves two small vectors per token per layer, a **key** and a **value**. Together they're the **KV cache**. Each new token adds its own and reads everyone else's.

The KV cache is what fills the GPU's memory. For this model:

| Storage format | Memory per token of context |
| --- | ---: |
| bf16 (the default, 2 bytes per number) | 160 KiB |
| int8 | 82.5 KiB |
| int4 | 42.5 KiB |
| K5:V4 (ours, described below) | 47.5 KiB |

A 100K-token conversation in bf16 needs 16 GB just for the cache, the whole card. After the weights (~8 GB), we have about 6–7 GB left for KV. The format decides how many tokens of context fit.

**What it means for you:** "context window" isn't only a model property; it's a memory budget on the serving side. A provider offering a 128K or 1M window is paying for that memory. It's also why long-context requests are often priced higher or queued.

---

## 4. Why long context gets slower

Attention compares each new token with every earlier one. At 500 tokens of context that's cheap; at 32,000 it isn't. For every output token, the model reads the whole KV cache as well as all the weights.

Real decode speeds from this project (with speculative decoding, section 6):

| Context length | Decode speed |
| ---: | ---: |
| ~5,600 tokens | 118 tok/s |
| ~17,000 tokens | 85 tok/s |
| ~32,000 tokens | 67 tok/s |

Prefill also grows faster than linearly with length. Every prompt token attends to all the tokens before it, so doubling the prompt more than doubles the attention work. A 32K prompt took 22 seconds to read on our first good build and 13.4 seconds after we rewrote the attention kernels. A 124K prompt takes about 2 minutes 20 seconds.

**What it means for you:** stuffing everything into the context has a real cost in latency, even when the price per token looks low. Retrieval (RAG) that sends 4K relevant tokens instead of 60K "just in case" tokens is faster on every single request.

---

## 5. Quantization: storing numbers with fewer bits

To fit more, you store numbers with fewer bits. That's **quantization**. The weights here are FP8 (8 bits instead of 16). The KV cache tried several formats:

| Format | Fits on the card | Quality |
| --- | ---: | --- |
| int8 | ~64K tokens | nearly identical to full precision |
| int4 | ~123K tokens | **noticeably worse** |
| K5:V4 (5-bit keys, 4-bit values) | ~110K tokens (147K after later savings) | close to int8 |

Two lessons from this part are worth knowing even if you never touch a GPU:

**Lossy compression can hurt answers in ways unit tests don't catch.** The int4 cache passed every kernel test (each stored number was within half a step of the original). But on a real task (a 50-case "golden set" of document-analysis questions), it hurt: finding contradictions dropped from 80% to 63%, and requirement recall dropped by 7 points. The error per number was tiny, but it compounds across 40 layers and thousands of tokens.

**Not all numbers are equally fragile.** A simulation showed that almost all the damage came from the keys, not the values. Giving keys 5 bits and values 4 bits fixed most of it for 12% more memory. That's how K5:V4 was chosen: by measuring first, then writing the code.

**What it means for you:** when a provider serves a "quantized" or "turbo" variant of a model, it can be measurably worse on your task even if the benchmarks look the same. If quality matters, test on your own cases. Also, the same model from two providers may not behave identically.

---

## 6. Speculative decoding: guessing ahead to beat the speed limit

Section 1 said 75 tokens/s is the ceiling. Yet this service writes 120–140 tokens/s. How?

The ceiling applies to one token per pass through the weights. But reading the weights for 1 token or for 16 tokens costs about the same: a step costs 16.3 ms for 1 token and 16.6 ms for 16. So the trick is to check several tokens per pass:

1. A tiny, fast **draft model** (here a single extra layer, the "EAGLE head") guesses the next few tokens.
2. The big model checks all the guesses in **one** pass, by computing what it would have written at each position.
3. Every guess that matches is kept for free. At the first wrong guess, the big model's own token is used instead, and drafting starts again from there.

The output is **exactly** what the big model would have written alone, token for token. The draft only changes speed, never content. This project checks that on every test: the speculative output must equal plain output.

How well it works depends on how predictable the text is. The share of guesses accepted ranged from about 75–80% on familiar styles to about 8% on novel creative prose. To guess better, the draft tries a small **tree** of alternatives instead of a single chain, and an adaptive policy turns drafting off when it isn't paying.

The draft model is trained by having the big model write lots of answers, then teaching the draft to predict them. One real finding: all our early drafts were trained on short conversations (≤ 4K tokens), so they got worse at long context. A new draft trained on 520 long, multi-file "agent reads" (3K–48K tokens) made decoding at 32K context **36% faster**.

**What it means for you:**
- Predictable output (code that mirrors the input, JSON with a fixed schema, summaries that quote the source) is often generated much faster than free-form writing. Many providers use speculative decoding behind the scenes.
- Speed per token can vary a lot by task, even on the same model and hardware.
- A well-done speed trick doesn't change the answer.

---

## 7. Prompt caching: don't re-read what you just read

When you send a follow-up message, the whole conversation is sent again. Without caching, the model would prefill all of it again. With **prompt caching** (also called prefix caching), the server keeps the KV cache from your last request. If the new request starts with exactly the same tokens, it skips straight to the new part.

Measured live, on a 25K-token prompt:

| Request | Time |
| --- | ---: |
| First time (cold) | 11.7 s |
| Same prompt again | 1.0 s |
| Same document, different question at the end | 0.7 s |
| Follow-up turn in the conversation | 0.8 s |

On a 56K-token prompt: 58 seconds cold, 0.6 seconds cached.

The catch is that matching is **exact, token by token, from the start**. One changed token early on (a timestamp, a different system prompt, tools listed in a different order) and everything after it must be re-read.

**What it means for you.** This is the most actionable section:
- Put stable content first and changing content last: system prompt, then tool definitions, then documents, then the conversation, then the new question.
- Don't put timestamps, request IDs or random values at the top of a prompt.
- Keep tool lists and their order fixed between calls.
- Many APIs bill cached input tokens at a large discount, for exactly this reason.

---

## 8. Serving several people at once

One user can't use the card's full speed: each step of one conversation reads all the weights for just a few tokens. But steps from different conversations can **share** one read of the weights. This server keeps 3 conversation slots and batches their steps together:

| Clients at once | Total output |
| ---: | ---: |
| 1 | ~100 tok/s |
| 2 | ~160 tok/s |
| 3 | ~211 tok/s |

Each slot needs KV memory, so there's a trade-off between more simultaneous users and longer context per user. Here, the 3 slots share a pool of 147,456 tokens. One conversation can use up to 128K of it, and the others wait for room if needed.

**What it means for you:** a provider's per-user speed and its total throughput are different numbers. Latency rises under load. Rate limits and "context length" limits are both about this same shared memory.

---

## 9. Determinism: same prompt, same answer?

People expect greedy decoding (temperature 0) to be perfectly repeatable. On real servers it often isn't, and this project measured why:
- Different code paths (a cached prefix vs a fresh read, batched vs alone) round numbers slightly differently.
- When two candidate tokens are nearly tied, a rounding difference in the last bit picks a different word, and the text diverges from there.

A surprising result: making the service perfectly deterministic made answer quality **worse** on the golden set, because retries and "ask three times and vote" stopped producing any variety. The harmful noise was the int4 cache's rounding, not the variation itself.

**What it means for you:** don't build anything that relies on byte-identical output from an LLM at temperature 0. If you need stable results, compare meaning, use structured output with validation, or vote over several samples.

---

## 10. Where the memory went, and why it mattered

At one point the card was full and a larger context window ran out of memory. A per-stage memory log found **1.7 GB of savings that had nothing to do with the model itself**:
- The token embedding table (784 MB) moved to normal system memory. Each token only reads one 8 KB row of it.
- Six identical scratch buffers became one shared buffer.
- A buffer sized for the worst-case prompt became a small reusable one.

That space went into the KV cache: 29% more context, and the move from a 64K to a 128K window.

**What it means for you:** "how much context can this hardware serve" depends heavily on the serving software, not just the GPU and the model. The same model on the same card can serve very different context lengths depending on how carefully the serving stack is built.

---

## 11. How we knew any of this was true

Most of the engineering effort went into measuring, not writing kernels:
- **Speed claims** came only from same-machine, same-prompt A/B comparisons.
- **Exactness** was checked by a "parity gate" at every service start (does the model still produce the reference tokens?) and by checking that speculative output equals plain output on every test.
- **Quality** was measured on real tasks (the golden set and needle-in-a-haystack tests up to 124K tokens), not just on proxy metrics.
- **Several apparent wins and losses were measurement errors.** Examples: a timer that didn't wait for the GPU, a test harness that swallowed the first reply, and a "−9% regression" that came from an eval set containing only 9 distinct prompts.

**What it means for you:** when you evaluate models or prompts, the same traps apply. Use enough distinct test cases, change one thing at a time, measure on your actual task, and be suspicious of any improvement you can't reproduce.

---

## Cheat sheet

| If you want... | Because... | Do this |
| --- | --- | --- |
| Faster responses | output tokens are sequential and slow | ask for shorter answers; prefer structured output |
| Cheaper, faster follow-ups | prompt caching matches exact prefixes | stable content first; no timestamps up top; fixed tool order |
| Fast long-document work | attention cost grows with context | retrieve the relevant parts instead of sending everything |
| Reliable quality | quantization and serving can shift behaviour | test the exact model and provider on your own cases |
| Repeatable output | near-ties flip on tiny rounding differences | validate structure and meaning, not exact bytes |
| Predictable speed | speculative decoding favours predictable text | expect code and JSON to stream faster than prose |

## Glossary

- **Token**: a piece of text (often part of a word) that the model reads or writes as one unit.
- **Weights / parameters**: the model's learned numbers; "8B" means 8 billion of them.
- **Layer**: one repeated block of the model; this one has 40.
- **Prefill**: reading the prompt, many tokens at once.
- **Decode**: writing the answer, one token at a time.
- **Memory-bandwidth-bound**: limited by how fast memory can be read, not by how fast the math runs.
- **Attention**: how each token looks back at earlier tokens to decide what matters.
- **KV cache**: the saved keys and values of earlier tokens, so attention doesn't recompute them.
- **Context window**: the maximum number of tokens (prompt plus answer) a request may use.
- **Quantization**: storing numbers with fewer bits to save memory and bandwidth.
- **Speculative decoding / draft model**: a small model guesses ahead and the big model verifies several guesses at once, with the same output, faster.
- **Acceptance rate**: the share of drafted tokens the big model agrees with.
- **Prompt / prefix caching**: reusing the KV cache of a prompt that starts with the same tokens.
- **Tensor cores**: GPU units built for fast bulk matrix math.
- **Greedy decoding / temperature 0**: always picking the single most likely next token.

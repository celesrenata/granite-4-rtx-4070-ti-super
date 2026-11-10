# K5:V4 KV cache (design and status)

Status: **DEPLOYED on gremlin-1 (2026-11-09, NixOS generation 793).** Branch `kv-k5v4` (commit 4993580, pushed, NOT yet
merged to `main`); the host flake input `granite4` is pinned to that branch/revision. It is opt-in
(`services.granite4-rtx.kv5 = true` / `G4_KV8=5`); nothing changes for other deploys. gremlin-1 runs kv5 with pool 114688.
Production verification after the switch: parity gate pass; engine `KV pool 114688 positions, main KV 5.20 GiB`; two concurrent
54.9K-token agents correct and resident (0.7 s resume); GPU peak 15,999 of 16,376 MiB; Menagerie golden set 300/300 valid.
Rollback: `nixos-rebuild switch --rollback` (generation 792, int4) or set `kv4 = true; kvPoolTokens = 122880;`.
Not yet done: merge to `main` (then point the host input back at the bare repo), a soak test under real agent traffic.

## Format (fixed; see `kernel_math.h`, "K5")

One row = 128 dims of one KV head at one position.

- K: 5-bit symmetric, `s = amax / 15`, `code = clamp(round_half_away(x / s), -15, 15) + 16` (1..31),
  `dequant = s * (code - 16)`. 80 bytes = 64 B low nibbles (byte-for-byte the int4 layout: dim 2j low, 2j+1 high)
  + 16 B high-bit plane (bit `d % 8` of byte `64 + d / 8`).
- V: unchanged Q4_0 (64 B + scale).
- Per position (40 layers x 8 KV heads): K 25 KiB + V 20 KiB + 2.5 KiB scales = **47.5 KiB**
  (int4 42.5, int8 82.5). Pool at the live ~5.0 GiB KV budget: **~110K positions**.

## Phases

| # | Phase | State |
| --- | --- | --- |
| 1 | Format + host twin (`k5_*` in `kernel_math.h`) + host tests incl. mutation check | done |
| 2 | Device twins (`dev::k5_*`) + quantize/append at all K+V write sites (decode, per-row, batched, fused rope+append, prefill) | done |
| 3 | Split-attention tile fills (decode / verify / rows kernel) and prefill attention read K5 + V4 | done |
| 4 | Cache/arena geometry (80-byte K rows, separate K/V buffer sizes), per-layer K offset, `kv_move`, context compatibility checks, `G4_KV8=5`, `gpu_ctx_is_k5v4` | done |
| 5 | GPU tests | done: 3 new K5 device tests + the full existing suite, all passing |
| 6 | Serve A/B on gremlin-1 | done (below) |

All new device code sits behind compile-time `K5` flags / `bits == 5` branches that default off, so the existing int4/int8/bf16
instantiations are unchanged. The draft head keeps its own int4 KV and is untouched.

### Results

- Build: `nix build .#engine-cuda` (sm_89) succeeds. Host suite 12/12.
- GPU, new tests (`G4_TEST_ONLY=k5v4`, runs next to a live service): device append is byte-identical to the host twin (0 bad K rows, 0 bad
  V bytes, 0 bad scales, neighbours untouched); decode attention matches bf16 attention over the dequantized cache (max delta 0.0039) and
  has ~29% lower error vs the original data than int4 (0.01155 vs 0.01638, realistic 1/sqrt(d) scale); prefill attention matches exactly
  (delta 0.0); prefill append matches the int4 path for q_rot and V and stays within one quantization step for K.
- GPU, full existing suite (service stopped, restored afterward): `all device kernel tests passed`, 0 FAIL lines (int4, int8, bf16, tree
  verify with KV move, shared arena sessions, paged attention).
- Serving (`G4_KV8=5`, EAGLE r12 k=4 on, 3 sessions x 64K): engine reports `KV pool 109568 positions, main KV 4.96 GiB` (= 47.5 KiB/pos as designed).
  Menagerie golden set, 50 cases x 6 repeats: schema-valid 100%, requirement recall 86.9%, contradiction recall 100% / false-positive 0%, risk
  accuracy 100% (model alone 100%), false PASS 0, p95 5.96 s. (int4: 99.7% / 80.3% / 63% / 90%; int8: 100% / 88.3% / 80% / 90%.)
- Long context: needle-in-haystack at 7K, 21K and 55.7K tokens, needle at 5% / 50% / 95%: 9 of 9 correct; prefill 55.7K = 58 s (1.04 ms/token;
  repo docs give ~1.05 ms/token for int4).
- Residency: with `GRANITE4_KV_POOL_TOKENS=114688` (5.20 GiB main KV), two concurrent 54.9K-token agents both answered correctly and BOTH stayed
  resident (resubmitting each took 0.7 s of prefix resume vs ~58 s for a re-prefill). Peak GPU memory 15,999 of 16,376 MiB. (Pool 109,568 is too
  small for two ~55K agents: 2 x 54.9K = 109.8K.)

### Not verified

- Behaviour under sustained real agent traffic, or over days. Tested for hours, mostly synthetic and Menagerie traffic.
- Memory headroom is ~377 MiB at pool 114,688; an agent workload with larger activations could OOM where the current config (pool 122,880 at int4) would not.
- The decode speed with K5 under pure chat load was not benchmarked (golden-set p50 0.75 s was unchanged).
- The NixOS module has no `kv5` option yet; testing used a systemd drop-in (`docs/performance/granite-serving-experiment.sh K` in Menagerie).

### Sites to change (found by reading the code; all currently assume one shared K/V format)

- Quantize + append, K and V written together (the K half becomes 5-bit): `forward_gpu.cu` ~199-211,
  ~441-454, ~563-575; `kernels_phase2.cu` ~497-510; `prefill_gemm.cu` ~492-505.
- Tile fill, K and V dequantized together into shared memory: `attention_split.cu` ~260-280 (kernel tile)
  and ~425-445 (second variant); dispatch/validation ~662-745 (`int8`/`int4` flags become a K/V format pair).
- Allocation and layout: `forward_gpu.cu` ~1430, ~3398 (row bytes, arena size); `kv_pool.cc` ~34-60
  (`bytes_per_token`); `infer_main.cc` ~1793 (`page_bytes`).
- Mode plumbing: `gpu_ctx_is_kv4/kv8` (`forward_gpu_api.h`), `G4_KV8` parsing, `frontend/config.py`, `flake.nix`.
- EAGLE draft KV has its own width (`G4_DRAFT_KV`) and is not touched.

### Risks / open questions (original, before implementation)

- **Not verified on a GPU.** Phases 2-4 can be compiled (the Nix build of `engine-cuda` targets sm_89) but not run
  on the dev machine. Phase 5 needs the card, which the live service occupies (15985 of 16376 MiB under load).
- **Speed is unknown.** K5 reads 25% more K bytes than int4 and adds a bit-plane load and bit-test per element.
  int4 prefill was measured slower than int8 in this repo; K5 could land anywhere between.
- **Quality is a proxy.** The simulation is teacher-forced KL/argmax on short prompts. The decision to proceed
  rests on it; confirm with the Menagerie golden set (phase 6) before deploying.
- Prefix/session resume (`serve_resume`) and the parity gate assume specific KV semantics; both need re-checking.

## EAGLE head retrain on K5:V4 features (r13k5, 2026-11-09): not adopted

Goal: recover the serving-style decode gap (EAGLE acceptance 0.145 with int4 vs 0.125 with K5:V4, r12 head).

- Features were re-dumped on the production RTX 4070 with the deployed engine (`G4_KV8=5`,
  `scripts/eagle/stage_r13k5_dump.sh`): serve_eval, serve_train3, chat_r12_eval, train2000, chat_r12_train,
  9.4M tokens in 43 min (production down for that window). The RTX 5090 was not used for dumping: on identical input
  the same binary's K5 features differ between the two cards (cosine 0.984 vs 0.9994 for int8).
- r13k5 was fine-tuned from r12 on the RTX 5090 (`scripts/eagle/train_remote_r13k5.sh`; same hyperparameters as r12,
  but only the three K5-dumped training sets).

Offline eval on K5 features (tokens per round at depth 6, higher is better):

| Head | chat_r12_eval (K5) | serve_eval (K5) |
| --- | ---: | ---: |
| r12 (trained on int8 features) | **2.186** | **2.455** |
| r13k5 | 2.154 | 2.389 |

r13k5 is worse on both. During training it dropped to 2.111 by step 500 and recovered only to 2.154, so the fine-tune
lost more (the r9/r11 chat sets and part of the base data were left out) than it gained from matching K5. r12 itself loses
only ~1% when moved from int8 to K5 features (2.206 at the end of its own training vs 2.186 here), so a perfectly matched
head has little left to win offline. The live serving-style gap is therefore probably not mainly a head-mismatch effect.
r12 stays deployed; r13k5 is kept at `heads/r13k5.safetensors` (gremlin) and `runs/r13k5/` (5090) but not used.
The K5 feature dumps are kept in `ds/k5/` on both hosts for future runs.

## Serving-style decode gap: policy and sample investigation (2026-11-09)

Two repeats per format, serving-style held-out prompts, EAGLE r12 k=4, fresh requests, `G4_SERVE_DEBUG=1`:

| Run | int4 | K5:V4 |
| --- | ---: | ---: |
| a | 153.6 tok/s, accept 0.145 | 140.0 tok/s, accept 0.125 |
| b | 153.5 tok/s, accept 0.145 | 140.0 tok/s, accept 0.125 |

- The result is exactly reproducible (deterministic greedy), so the -9% is not run-to-run noise.
- The adaptive draft policy is not involved: both formats kept drafting on for every request (0 switches to plain).
- The 24-prompt set holds only **9 distinct prompts** (repeated 4, 4, 3, 3, 2, 2, 2, 2, 2 times). Per distinct prompt, K5 is slower on 3 (-20%, -12%, -14%), equal on 3, and faster on 3 (up to +10%). The weighted -9% is dominated by the single largest 4x-repeated prompt (176 vs 140 tok/s).
- None of the 24 outputs are identical between formats: each format follows its own greedy continuation, and the draft head finds some continuations easier to predict than others.

Conclusion: the serving-style gap is per-content variance between two different greedy continuations measured on a small, duplicated set, not a systematic K5 cost. With the other families (chat -1.6% over 30 prompts, novel +0.7%, long +3.4%) and r12 losing only ~1% offline when moved to K5 features, speculative decode speed under K5:V4 is effectively equal to int4. No remaining K5-specific lever was found; a faster decode would need general head or engine improvements.

## EAGLE head retrain, full mix (r14k5, 2026-11-09): not adopted

r14k5 retrains r12's full five-set mix (chat_r9/r11/r12_train, serve_train3, train2000) on K5:V4 features dumped on gremlin-1 by the production engine. It is fine-tuned from r12 with r12's exact recipe (`scripts/eagle/train_remote_r14k5.sh`, 5589 steps, about 57 min on the 5090).

| Head | chat_r12_eval (K5) | serve_eval (K5) |
| --- | ---: | ---: |
| r12 | **2.186** | **2.455** |
| r13k5 (3 of the 5 sets) | 2.154 | 2.389 |
| r14k5 (full mix) | 2.182 | 2.431 |

With the full mix, the loss r13k5 showed is gone, but r14k5 still does not beat r12. Matching the head to K5 features has nothing left to win, so r12 stays deployed. Further head gains would have to come from elsewhere: more data, a deeper head, or a different tree.

## Tensor-core split-KV attention (`G4_ATTN_MMA`, default on, 2026-11-09)

`granite4-attn-bench` (new; serve shapes, pool 114688, maxlen 65536, 14 rows, ms per 40 layers) showed why verify attention was slow. On the scalar kernels a 14-row verify took the same time in every KV format, about 11% of the K5:V4 bytes-read floor, so it was compute bound. K5 decode at 16K reached only 33% even though its 20 MB working set fits in the 48 MB L2, so it was bound by the per-element dequantize work.

`attn_mma_kernel` (in `attention_split.cu`) replaces the scalar kernels for every attention call: decode, chain verify, multi-session rows and draft-tree rows.
- Layout: one block per (KV head, split, group of up to 8 rows). The query rows are (row, query head) pairs, giving two m16 tiles. Each 64-key tile computes S = QKᵀ and O += PV with `mma.m16n8k16` bf16 at f32 accumulate. The online softmax is in f32 and the logits are rounded to bf16 as before.
- V fragments: `ldmatrix.trans` from a row-major tile.
- Pipelining: the next tile's cache loads are issued before the current tile is attended. The split's page-table rows are staged in shared memory once per block.
- Quantized formats: the tiles hold the integer codes, which are exact in bf16. The K scale multiplies the logit column and the V scale folds into P, so a code is converted with one `0x4300|n` OR and one `fma.bf16x2` per pair, and no per-element multiply or bf16 rounding remains. K is therefore used exactly instead of being rounded to bf16 after scaling. Mean |err| vs the original values is 0.01152 (was 0.01155). The max |delta| vs the old dequantized-bf16 reference grew to 0.0117, and that reference no longer matches the arithmetic.
- Exactness: an MMA output row depends only on its own query row, and masked keys contribute exactly 0. A row's result therefore does not depend on which rows share its block. Decode, grouped verify and tree verify stay bit-identical (`attn_split rows/groups`, `decode multi`, `tree verify` and `shared arena` device tests, all four formats). Tree tiles past the shared prefix run once per row with that row's own keys, and only that row is updated.
- Fallback: `G4_ATTN_MMA=0` restores the scalar kernels. The setting is per process, so decode and verify always share one kernel.

K5:V4 results (scalar → MMA):

| Case | 4K | 16K | 32K | 55K |
| --- | ---: | ---: | ---: | ---: |
| tree verify (14 rows, production) | 5.56 → 2.36 | 22.96 → 8.54 | 46.12 → 17.63 | 77.67 → 30.15 |
| decode | 0.87 → 0.80 | 3.35 → 3.21 | 7.33 → 7.11 | 11.96 → 11.50 |
| one row per block (14 rows) | 9.40 → 8.23 | 41.56 → 38.20 | 93.43 → 87.55 | 156.70 → 146.58 |

KV4 and KV8 gain about the same. bf16 grouped verify is 1.9–2.9× faster, while bf16 decode is 3–9% slower at ≤4K (about 0.07 ms per token; bf16 KV is not the serving format).

Live results on gremlin-1 (`G4_MULTI_PROFILE`, r12 k=4 tree verify, 14 rows, ms per verify step):

| ctx | attn before | attn after | step before → after |
| ---: | ---: | ---: | ---: |
| 541 | 1.50 | 1.66 | 17.0 → 17.2 |
| 4K | 6.38 | 2.63 | 22.2 → 18.5 |
| 16K | 22.21 | 7.31 | 38.8 → 23.9 |
| 32K | – | 14.77 | – → 31.1 |

On the held-out families (`kv_accept_ab.py`, K5, r12 k=4), speculative output equals plain greedy for every request, and parity passes (teacher-forced, K5 and bf16, MMA on and off). Plain decode is unchanged at short context (68 tok/s, weight-bound). The greedy text differs from the scalar build because the attention arithmetic changed: 4 of 61 plain outputs are identical. Tok/s on chat/novel/serving is within the per-content variance described above (137.2 / 136.4 / 135.6 vs 134.7 / 138.0 / 140.3). The long family (10–13K-token prompts) goes from 30.8 to 34.2 tok/s including prefill.

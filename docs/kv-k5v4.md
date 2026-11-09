# K5:V4 KV cache (design and status)

Status: **phases 1-6 implemented and measured on gremlin-1 (2026-11-09); NOT deployed.** The change lives on
branch `kv-k5v4` (uncommitted working tree). It is opt-in (`G4_KV8=5`); nothing changes unless that is set.
The live service was restored to its original int4 config after testing. Not yet done: NixOS module option
(`kv5`), upstreaming, and a soak test under real agent traffic.

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

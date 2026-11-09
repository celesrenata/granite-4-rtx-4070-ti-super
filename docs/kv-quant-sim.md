# KV quantization simulation (why K5:V4)

Reference-model study, not a GPU measurement. The simulator (`kv_quant_sim.py`, in the Menagerie
repo at `scripts/kv-quant-sim/`) runs the bf16 checkpoint with K (post-RoPE) and V fake-quantized
exactly like `kernel_math.h` (symmetric, one scale per 128-element row, `scale = amax / qmax`,
`qmax = 2^(bits-1) - 1`), teacher-forcing 50 real completions (4,607 positions) and comparing
each K:V width pair with the unquantized model. Raw data: Menagerie `docs/performance/kv-quant-sim-result.json`.

| K:V | KiB/pos | Pool in ~5.0 GiB | Mean KL | Top-1 flip |
| --- | ---: | ---: | ---: | ---: |
| 8:8 | 82.5 | 63,550 | 0.0019 | 0.98% |
| 8:4 | 62.5 | 83,886 | 0.0149 | 3.15% |
| 6:4 | 52.5 | 99,864 | 0.0293 | 4.67% |
| 5:5 | 52.5 | 99,864 | 0.0752 | 6.75% |
| **5:4** | **47.5** | **110,376** | **0.0840** | **7.08%** |
| 4:8 | 62.5 | 83,886 | 0.2833 | 13.94% |
| 4:4 (current) | 42.5 | 123,361 | 0.2877 | 14.59% |

K carries almost all of the int4 damage (4:8 is as bad as 4:4; 8:4 removes 95% of it). K5:V4 cuts the
error 3.4x versus the current int4 cache for +5 KiB/position, and its pool (110K positions) still holds
two ~55K sessions. Limits: ~300-450 token prompts only, proxies (KL, argmax flips) not task accuracy,
no speed data.

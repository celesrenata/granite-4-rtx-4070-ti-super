# KV8: full int8 (Q8_0) KV cache across all forward paths

Goal: store the KV cache as int8 with a per-row scale (Q8_0), quantize-on-append,
dequant-on-load inside attention. Then measure generation quality (token match +
logits/hidden divergence) and the memory/DRAM-bandwidth saving vs the bf16 cache.

Decision (user): full int8 store, all paths. This model is for reading, so report
generation-quality metrics, not just memory.

## Files
- engine/src/forward_gpu.cu   GpuCtx + alloc/free + LayerView + 3 forward paths + decode append
- engine/src/kernels_phase2.cu  batched verify append + fused GQA row
- engine/src/attention_split.cu  split-KV attention (decode default)
- engine/src/prefill_gemm.cu    prefill append (rope_split_append) + prefill attention
- engine/src/eagle_gpu.cu       eagle head append/attention
- engine/tests/test_kernels.cu  device parity tests
- scripts/ accuracy harness (new)

## Plan
1. GpuCtx: kv8 flag + 4 int8/scale buffers; LayerView int8 ptrs
2. int8 append kernels: decode, batched-verify, prefill
3. int8 attention: split-KV (decode/verify), fused GQA (fallback), prefill
4. route all 3 forward paths + eagle head
5. tests: device append parity + dequant parity
6. harness: bf16 vs int8 same-prompt token match + divergence
7. build+run on gremlin-1

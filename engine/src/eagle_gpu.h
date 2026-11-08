// EAGLE-style draft head on the GPU (eagle_gpu.cu; trained by
// scripts/eagle/train_eagle.py, exported by scripts/eagle/export_head.py).
#ifndef GRANITE4_EAGLE_GPU_H
#define GRANITE4_EAGLE_GPU_H

#include <cstddef>
#include <cstdint>
#include <string>

#include "forward_gpu_api.h"
#include "model_config.h"

namespace granite4 {

struct EagleHead;

// Load a head (safetensors: fc.{weight,bias}, ln1/ln2/lnf.weight,
// q/k/v/o/gate/up/down.weight, draft_vocab int32) for `target`, whose packed
// embedding and FP8 lm_head it shares. max_pos bounds the head's KV cache.
// Returns nullptr with *err on failure.
//
// kv_rows == 0 (default): the head owns a private [max_pos] draft cache.
// kv_rows > 0: the draft KV is ONE shared arena of kv_rows positions per
// head - the same row space as the target's shared KV arena - addressed
// through `target`'s row table, so the single pool serves every session. This
// head then serves `target`; eagle_fork adds the other sessions.
EagleHead* eagle_load(const std::string& path, GpuCtx* target, const ModelConfig& cfg,
                      const ForwardSpec& spec, int max_pos, std::string* err,
                      int kv_rows = 0);
// A per-session head on `base`'s weights and draft-KV arena: own scratch and
// watermark, rows addressed through `target`'s row table (a context on the
// same arena as base's target). `base` must stay alive and must have been
// loaded with kv_rows > 0. Free forks before the base.
EagleHead* eagle_fork(const EagleHead* base, GpuCtx* target, std::string* err);
// Bytes of the draft-KV pool (shared by the base and all forks).
std::size_t eagle_kv_bytes(const EagleHead* h);
void eagle_free(EagleHead* h);

// Head entries [pos0, pos0 + rows): entry j reads token d_tokens[j + 1] and
// feature row d_feat[j - pos0] ([rows][H] bf16, device) and writes its
// predicted feature to d_feat_out[j - pos0] (may alias d_feat). pos0 must be
// the current watermark (eagle_written). With d_draft_out, the last row's
// draft token (draft-vocab argmax mapped to a target id) is written there.
// Enqueued on the per-thread stream; nothing is synchronized.
int eagle_forward(EagleHead* h, const std::int32_t* d_tokens, int pos0, int rows,
                  const std::uint16_t* d_feat, std::uint16_t* d_feat_out,
                  std::int32_t* d_draft_out);

// Drop head entries >= written.
void eagle_truncate(EagleHead* h, int written);
int eagle_written(const EagleHead* h);
// Weight bytes streamed per draft step (FP8 head + draft-vocab lm_head).
std::size_t eagle_device_bytes(const EagleHead* h);

}  // namespace granite4

#endif  // GRANITE4_EAGLE_GPU_H

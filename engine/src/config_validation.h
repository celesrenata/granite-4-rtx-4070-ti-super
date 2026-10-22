// Config validation for the Granite 4.2 8B engine.
//
// Two jobs:
//  1. is_default() - does a candidate config match the target model's spec?
//  2. structural checks - e.g. head count divides hidden size, GQA ratio.
//
// The C ABI (capi.cc) exposes these so the Python frontend can reject a
// misconfigured checkpoint before doing any GPU work.
#ifndef GRANITE4_CONFIG_VALIDATION_H
#define GRANITE4_CONFIG_VALIDATION_H

#include "model_config.h"

#include <string>

namespace granite4 {

// True if cfg is exactly kModelConfig.
bool matches_target_model(const ModelConfig& cfg);

// True if cfg is structurally consistent for a transformer, whatever the
// concrete values (positive dims, GQA divisibility, head_dim * heads == hidden).
bool is_structurally_consistent(const ModelConfig& cfg);

// Human-readable description of every deviation of cfg from the target spec.
// Empty string when cfg matches exactly.
std::string describe_deviations(const ModelConfig& cfg);

}  // namespace granite4

#endif  // GRANITE4_CONFIG_VALIDATION_H

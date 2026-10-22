#include "config_validation.h"

#include <sstream>
#include <utility>
#include <vector>

namespace granite4 {

bool matches_target_model(const ModelConfig& cfg) { return cfg == kModelConfig; }

bool is_structurally_consistent(const ModelConfig& cfg) {
  if (cfg.num_layers == 0 || cfg.hidden_size == 0 || cfg.intermediate_size == 0) return false;
  if (cfg.num_attention_heads == 0 || cfg.num_key_value_heads == 0 || cfg.head_dim == 0) return false;
  if (cfg.num_attention_heads % cfg.num_key_value_heads != 0) return false;
  if (cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size) return false;
  if (cfg.vocab_size == 0) return false;
  if (cfg.max_context_length == 0) return false;
  return true;
}

namespace {
void field(std::vector<std::string>& out, const char* name, std::size_t expected, std::size_t got) {
  if (expected != got) {
    std::ostringstream os;
    os << name << ": expected " << expected << ", got " << got;
    out.push_back(os.str());
  }
}
}  // namespace

std::string describe_deviations(const ModelConfig& cfg) {
  std::vector<std::string> issues;
  field(issues, "num_layers", kModelConfig.num_layers, cfg.num_layers);
  field(issues, "hidden_size", kModelConfig.hidden_size, cfg.hidden_size);
  field(issues, "intermediate_size", kModelConfig.intermediate_size, cfg.intermediate_size);
  field(issues, "num_attention_heads", kModelConfig.num_attention_heads, cfg.num_attention_heads);
  field(issues, "num_key_value_heads", kModelConfig.num_key_value_heads, cfg.num_key_value_heads);
  field(issues, "head_dim", kModelConfig.head_dim, cfg.head_dim);
  field(issues, "vocab_size", kModelConfig.vocab_size, cfg.vocab_size);
  field(issues, "max_context_length", kModelConfig.max_context_length, cfg.max_context_length);
  if (!issues.empty() || !is_structurally_consistent(cfg)) {
    if (issues.empty()) issues.push_back("structurally inconsistent");
    std::string joined;
    for (std::size_t i = 0; i < issues.size(); i++) {
      if (i) joined += "; ";
      joined += issues[i];
    }
    return joined;
  }
  return std::string();
}

}  // namespace granite4

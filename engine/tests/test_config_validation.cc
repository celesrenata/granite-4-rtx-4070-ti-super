// Host-only test of the model constants, config validation and the C ABI.
// No CUDA required; run via ctest after a host-only build.
#include "capi.h"
#include "config_validation.h"
#include "model_config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

int g_failures = 0;

void expect(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

void expect_eq(long a, long b, const char* what) {
  if (a != b) {
    std::printf("FAIL: %s (got %ld, want %ld)\n", what, a, b);
    ++g_failures;
  }
}

}  // namespace

int main() {
  using namespace granite4;

  // --- constants: the Granite 4.2 8B spec ---
  expect(kModelConfig.num_layers == 40, "num_layers == 40");
  expect(kModelConfig.hidden_size == 4096, "hidden_size == 4096");
  expect(kModelConfig.intermediate_size == 12800, "intermediate_size == 12800");
  expect(kModelConfig.num_attention_heads == 32, "num_attention_heads == 32");
  expect(kModelConfig.num_key_value_heads == 8, "num_key_value_heads == 8");
  expect(kModelConfig.head_dim == 128, "head_dim == 128");
  expect(kModelConfig.vocab_size == 100352, "vocab_size == 100352");
  expect(kModelConfig.max_context_length == 131072, "max_context_length == 131072");
  expect(kCudaArchMajor == 8 && kCudaArchMinor == 9, "compute capability 8.9");
  expect(std::strcmp(kModelId, "granite-4.2-8b") == 0, "model id string");
  expect(std::strcmp(kGpuTarget, "sm_89") == 0, "gpu target string");

  // --- validation ---
  expect(matches_target_model(kModelConfig), "default matches target");
  expect(is_structurally_consistent(kModelConfig), "default is structurally consistent");
  expect(describe_deviations(kModelConfig).empty(), "no deviations for default");

  ModelConfig bad = kModelConfig;
  bad.num_layers = 32;
  expect(!matches_target_model(bad), "mutated config does not match target");
  std::string d = describe_deviations(bad);
  expect(d.find("num_layers") != std::string::npos, "deviation names the field");
  expect(d.find("32") != std::string::npos, "deviation includes the bad value");

  ModelConfig gqa_bad = kModelConfig;
  gqa_bad.num_key_value_heads = 5;  // 32 % 5 != 0
  expect(!is_structurally_consistent(gqa_bad), "GQA ratio 32/5 is inconsistent");
  expect(!describe_deviations(gqa_bad).empty(), "deviations reported for GQA mismatch");

  ModelConfig dim_bad = kModelConfig;
  dim_bad.head_dim = 120;  // 32 * 120 != 4096
  expect(!is_structurally_consistent(dim_bad), "head_dim*heads != hidden is inconsistent");

  // --- C ABI: metadata ---
  granite4_model_info info{};
  granite4_get_model_info(&info);
  expect(std::strcmp(info.model_id, "granite-4.2-8b") == 0, "ABI model id");
  expect_eq(info.num_layers, 40, "ABI num_layers");
  expect_eq(info.hidden_size, 4096, "ABI hidden_size");
  expect_eq(info.intermediate_size, 12800, "ABI intermediate_size");
  expect_eq(info.num_attention_heads, 32, "ABI num_attention_heads");
  expect_eq(info.num_key_value_heads, 8, "ABI num_key_value_heads");
  expect_eq(info.head_dim, 128, "ABI head_dim");
  expect_eq(info.vocab_size, 100352, "ABI vocab_size");
  expect_eq(info.max_context_length, 131072, "ABI max_context_length");
  expect(info.gpu_target != nullptr && std::strcmp(info.gpu_target, "sm_89") == 0, "ABI gpu target");

  // --- C ABI: status placeholder ---
  granite4_status_info st{};
  granite4_status(&st);
  expect(st.status == granite4_unimplemented, "Phase 0 status is unimplemented");
  expect(st.status_text != nullptr && std::strcmp(st.status_text, "unimplemented") == 0,
         "status text matches code");
  expect(st.message != nullptr, "status message present (may be empty)");

  // --- C ABI: NULL tolerance ---
  granite4_get_model_info(nullptr);  // must not crash
  granite4_status(nullptr);         // must not crash

  if (g_failures == 0) {
    std::printf("PASS: all engine config/ABI checks\n");
    return 0;
  }
  std::printf("%d check(s) failed\n", g_failures);
  return 1;
}

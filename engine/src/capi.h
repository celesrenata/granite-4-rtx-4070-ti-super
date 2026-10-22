// C ABI of the Granite 4.2 8B engine.
//
// Phase 0 (scaffold): metadata and status only. There is no inference.
// granite4_status always reports inference as unavailable; every function
// that would need a GPU reports granite4_unavailable rather than pretending.
//
// All functions are process-safe and side-effect free. Strings returned by
// pointer-out parameters point at static storage and do not need freeing.
#ifndef GRANITE4_CAPI_H
#define GRANITE4_CAPI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes returned by granite4_status. */
typedef enum {
  /* Server reachable, API responsive. */
  granite4_status_ok = 0,
  /* Library loaded but inference is not implemented (Phase 0 scaffold). */
  granite4_unimplemented = 1,
  /* Library loaded, but no CUDA device is present or usable. */
  granite4_no_device = 2,
  /* Library failed to load or has an unexpected ABI. */
  granite4_unavailable = 3,
} granite4_status_code;

/* Metadata of the target model. Values are the spec, not a loaded state:
 * nothing is loaded until P1, and P1 must re-validate these against the
 * checkpoint before they become runtime truth. */
typedef struct {
  const char* model_id;
  long num_layers;
  long hidden_size;
  long intermediate_size;
  long num_attention_heads;
  long num_key_value_heads;
  long head_dim;
  long vocab_size;
  long max_context_length;
  /* Target compute capability, "MAJOR.MINOR". */
  const char* gpu_target;
} granite4_model_info;

/* Library status. The out-parameters receive NULL-safe, static strings;
 * `message` is empty when status is granite4_status_ok. */
typedef struct {
  granite4_status_code status;
  /* 1 when a CUDA device is present (queried at call time). */
  int cuda_device_present;
  const char* status_text;
  const char* message;
} granite4_status_info;

/* Returns the target model's constants. Stable across calls; the struct is
 * copied by value, no allocation. */
void granite4_get_model_info(granite4_model_info* out);

/* Reports the library status. In Phase 0 the status is always
 * granite4_unimplemented - the API responds, but no inference exists yet.
 * `out` must not be NULL. */
void granite4_status(granite4_status_info* out);

/* Result of a checkpoint metadata (safetensors header) validation.
 * `diagnostic` is a bounded "code: message" string, empty on success,
 * never longer than 255 bytes. */
typedef struct {
  int ok;
  long tensor_count;
  char diagnostic[256];
} granite4_safetensors_result;

/* Validates the bounded metadata prefix (8-byte little-endian header length
 * plus JSON header) of the safetensors file at `path`. Tensor data is never
 * read. `path` must be a NUL-terminated string; `out` must be non-NULL.
 * Phase-1 host-side API: no CUDA, no inference. */
void granite4_validate_safetensors(const char* path,
                                   granite4_safetensors_result* out);

#ifdef __cplusplus
}
#endif

#endif /* GRANITE4_CAPI_H */

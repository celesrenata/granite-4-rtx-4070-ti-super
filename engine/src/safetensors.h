// Phase-1 (host-only) Safetensors metadata foundation for the Granite 4.2 8B
// engine.
//
// Bounded, stdlib-only parser for the safetensors container format:
//
//   [8-byte little-endian u64: header length N] [N bytes: JSON header] [data]
//
// The JSON header is a flat object mapping tensor names to
// {"dtype": <string>, "shape": [<int>, ...], "data_offsets": [start, end]}
// entries; the optional "__metadata__" member (a JSON object) is accepted
// and ignored.
//
// Guarantees:
//  * The header length N is validated before any allocation: N == 0 and
//    N > kMaxHeaderBytes (64 MiB) are rejected; a file shorter than
//    8 + N is rejected as truncated.
//  * JSON parsing is depth-bounded and rejects malformed syntax, duplicate
//    object keys, invalid escapes / UTF-8, unescaped control characters,
//    trailing content after the root value, and non-integer numbers where
//    integers are required.
//  * Tensor entries are validated for missing / unknown fields, invalid
//    dtype (only the canonical safetensors dtype names are accepted),
//    invalid shape (non-integer or negative dimensions, product overflow),
//    invalid data_offsets (not exactly two non-ascending integers),
//    shape/dtype/byte-range consistency (end - start == numel * itemsize),
//    and out-of-bounds data ranges relative to the total file size.
//  * The result is an immutable manifest (TensorManifest, populated only by
//    the parse functions, to be treated as const afterwards) with a const
//    verification API (verify / describe_issues) and an optional
//    expected-name delta (diff_expected_names). This module does NOT
//    hardcode any Granite checkpoint tensor names.
//
// Metadata only: no weight data is read, no execution, no CUDA.
#ifndef GRANITE4_SAFETENSORS_H
#define GRANITE4_SAFETENSORS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace granite4 {

// Upper bound on the JSON header length. Headers above this are rejected as
// unreasonable before any allocation proportional to N happens.
inline constexpr std::size_t kMaxHeaderBytes = 64u * 1024u * 1024u;

// One tensor record from the header. Populated by the parse functions;
// treat as immutable (the manifest contract forbids mutation after parse).
struct TensorEntry {
  std::string name;
  std::string dtype;  // canonical safetensors dtype name, e.g. "BF16"
  std::vector<int64_t> shape;
  int64_t offset_begin = 0;  // byte offset within the data region, inclusive
  int64_t offset_end = 0;    // byte offset within the data region, exclusive
  int64_t numel = 0;         // product(shape), overflow-checked
  int64_t nbytes = 0;        // offset_end - offset_begin
};

// Immutable tensor manifest parsed from a safetensors header.
//
// Populated exclusively by parse_safetensors / parse_safetensors_file;
// callers must treat a parsed manifest as const. verify() re-checks every
// invariant (it is cheap insurance against future code that constructs or
// patches manifests manually) and is part of the const API.
struct TensorManifest {
  int64_t header_bytes = 0;      // N from the 8-byte little-endian prefix
  int64_t data_region_bytes = 0; // total file size - 8 - N (>= 0 after parse)
  bool has_metadata = false;     // true when a "__metadata__" member was present
  std::vector<TensorEntry> tensors;  // in header order

  // O(n) name lookup; nullptr when absent. "__metadata__" is never a tensor.
  const TensorEntry* find(const std::string& name) const;

  // True when all invariants hold (unique names, known dtype, non-negative
  // dims, stored numel consistent with shape, ascending offsets within the
  // data region, byte range consistent with shape and dtype).
  bool verify() const;

  // Human-readable list of invariant violations; empty when verify() is true.
  std::string describe_issues() const;

  // Optional expected-name delta (callers supply the expected names; this
  // module never invents checkpoint tensor names):
  //  * missing   = expected names absent from the manifest (expected order,
  //                duplicates collapsed to first occurrence);
  //  * unexpected = manifest names not in the expected set (header order).
  // nullptr out-parameters are skipped.
  void diff_expected_names(const std::vector<std::string>& expected,
                           std::vector<std::string>* missing,
                           std::vector<std::string>* unexpected) const;

  // Sum of tensor byte ranges. Returns -1 when the sum overflows int64.
  int64_t total_bytes() const;
};

// Parse a safetensors byte buffer in memory.
//
// `data` must contain the 8-byte little-endian header length followed by N
// header bytes; `data_size` is the number of available bytes (>= 8 + N
// required, else "truncated"). `file_size` is the total file size
// (>= data_size); out-of-bounds data ranges are checked against it. For a
// pure in-memory buffer, pass file_size == data_size.
//
// On success: *out is populated and true is returned. On failure: false is
// returned and *error (when non-null) receives a bounded diagnostic of the
// form "code: message" (at most 255 characters, safe to copy into a fixed
// C buffer).
bool parse_safetensors(const char* data, std::size_t data_size,
                       std::size_t file_size, TensorManifest* out,
                       std::string* error);

// Read only the bounded metadata prefix (exactly 8 + N bytes) of the file
// at `path` and parse it; tensor data is never read. Same error contract as
// parse_safetensors.
bool parse_safetensors_file(const std::string& path, TensorManifest* out,
                            std::string* error);

// Byte size of one element for the canonical safetensors dtypes
// (F64, F32, F16, BF16, I64, I32, I16, I8, U8, F8_E4M3, F8_E5M2).
// Returns false (and leaves *item_size untouched) for any other name.
bool dtype_item_size(const std::string& dtype, int64_t* item_size);

}  // namespace granite4

#endif  // GRANITE4_SAFETENSORS_H

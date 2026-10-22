// Phase-1 (host-only) Safetensors metadata parser.
//
// Bounded, stdlib-only. See safetensors.h for the contract. This file
// intentionally does not read tensor data, execute weights, touch CUDA, or
// hardcode any Granite checkpoint tensor names.
#include "safetensors.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <utility>

namespace granite4 {
namespace {

constexpr int64_t kInt64Max = INT64_MAX;
constexpr int64_t kInt64Min = INT64_MIN;

// --- error helpers ---------------------------------------------------------

// Bounded diagnostic: "code: message", never longer than 255 bytes.
std::string make_error(const char* code, const std::string& message) {
  std::string out = code;
  out += ": ";
  out += message;
  constexpr std::size_t kMaxErrorBytes = 255;
  if (out.size() > kMaxErrorBytes) out.resize(kMaxErrorBytes);
  return out;
}

// --- little-endian decode --------------------------------------------------

void le64(const char* p, uint64_t* out) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | static_cast<uint8_t>(p[i]);
  *out = v;
}

// --- overflow-checked arithmetic -------------------------------------------

bool safe_mul(int64_t a, int64_t b, int64_t* out) {
  if (a == 0 || b == 0) {
    *out = 0;
    return true;
  }
  if (a < 0 || b < 0) return false;
  if (a > kInt64Max / b) return false;
  *out = a * b;
  return true;
}

bool safe_add(int64_t a, int64_t b, int64_t* out) {
  if (b > 0 && a > kInt64Max - b) return false;
  if (b < 0 && a < kInt64Min - b) return false;
  *out = a + b;
  return true;
}

// --- bounded JSON parser ---------------------------------------------------
//
// Minimal JSON value parser with depth and syntax bounds. Accepts the subset
// needed for a safetensors header: objects, arrays, strings, strict integers,
// and the literals true/false/null. Floating-point / exponent numbers,
// nesting deeper than kMaxJsonDepth, malformed escapes, unescaped control
// characters and trailing content are all hard errors. Object keys are kept in
// document order (std::vector) so the caller can detect duplicates and
// preserve header tensor order.

constexpr int kMaxJsonDepth = 128;

struct JsonValue {
  enum class Kind { Null, Bool, Int, String, Array, Object } kind = Kind::Null;
  bool bool_value = false;
  int64_t int_value = 0;
  std::string str;
  std::vector<JsonValue> arr;
  std::vector<std::pair<std::string, JsonValue>> obj;

  const JsonValue* get(const std::string& key) const {
    for (const auto& kv : obj) {
      if (kv.first == key) return &kv.second;
    }
    return nullptr;
  }
};

struct JsonParser {
  const char* p = nullptr;
  const char* end = nullptr;
  int depth = 0;
  bool error = false;
  std::string err;

  void fail(const char* msg) {
    if (!error) {
      error = true;
      err = msg;
    }
  }

  void skip_ws() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
  }

  bool at(const char* lit) const {
    const std::size_t n = std::strlen(lit);
    return static_cast<std::size_t>(end - p) >= n && std::memcmp(p, lit, n) == 0;
  }

  JsonValue parse_value() {
    skip_ws();
    if (p >= end) {
      fail("unexpected end of JSON input");
      return {};
    }
    char c = *p;
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') {
      JsonValue v;
      v.kind = JsonValue::Kind::String;
      parse_string(&v.str);
      return v;
    }
    if (c == '-' || (c >= '0' && c <= '9')) return parse_int();
    if (at("true")) {
      p += 4;
      JsonValue v;
      v.kind = JsonValue::Kind::Bool;
      v.bool_value = true;
      return v;
    }
    if (at("false")) {
      p += 5;
      JsonValue v;
      v.kind = JsonValue::Kind::Bool;
      v.bool_value = false;
      return v;
    }
    if (at("null")) {
      p += 4;
      JsonValue v;
      v.kind = JsonValue::Kind::Null;
      return v;
    }
    fail("expected a JSON value");
    return {};
  }

  JsonValue parse_object() {
    JsonValue v;
    v.kind = JsonValue::Kind::Object;
    if (++depth > kMaxJsonDepth) {
      --depth;
      fail("JSON nesting depth exceeds bound");
      return v;
    }
    ++p;  // '{'
    skip_ws();
    if (p < end && *p == '}') {
      ++p;
      --depth;
      return v;
    }
    for (;;) {
      skip_ws();
      if (p >= end) {
        fail("unterminated object");
        break;
      }
      if (*p != '"') {
        fail("expected a string key in object");
        break;
      }
      JsonValue key;
      key.kind = JsonValue::Kind::String;
      parse_string(&key.str);
      if (error) {
        --depth;
        return v;
      }
      skip_ws();
      if (p >= end || *p != ':') {
        fail("expected ':' after object key");
        break;
      }
      ++p;
      JsonValue val = parse_value();
      v.obj.emplace_back(std::move(key.str), std::move(val));
      skip_ws();
      if (p >= end) {
        fail("unterminated object");
        break;
      }
      if (*p == ',') {
        ++p;
        continue;
      }
      if (*p == '}') {
        ++p;
        break;
      }
      fail("expected ',' or '}' in object");
      break;
    }
    --depth;
    return v;
  }

  JsonValue parse_array() {
    JsonValue v;
    v.kind = JsonValue::Kind::Array;
    if (++depth > kMaxJsonDepth) {
      --depth;
      fail("JSON nesting depth exceeds bound");
      return v;
    }
    ++p;  // '['
    skip_ws();
    if (p < end && *p == ']') {
      ++p;
      --depth;
      return v;
    }
    for (;;) {
      v.arr.push_back(parse_value());
      skip_ws();
      if (p >= end) {
        fail("unterminated array");
        break;
      }
      if (*p == ',') {
        ++p;
        continue;
      }
      if (*p == ']') {
        ++p;
        break;
      }
      fail("expected ',' or ']' in array");
      break;
    }
    --depth;
    return v;
  }

  JsonValue parse_int() {
    JsonValue v;
    v.kind = JsonValue::Kind::Int;
    bool neg = false;
    if (*p == '-') {
      neg = true;
      ++p;
    }
    if (p >= end || *p < '0' || *p > '9') {
      fail("invalid integer literal");
      return v;
    }
    int64_t n = 0;
    while (p < end && *p >= '0' && *p <= '9') {
      const int digit = *p - '0';
      if (n > (kInt64Max - digit) / 10) {
        fail("integer literal out of range");
        return v;
      }
      n = n * 10 + digit;
      ++p;
    }
    // A following '.', 'e' or 'E' means this was really a float / exponent,
    // which the safetensors header forbids.
    if (p < end && (*p == '.' || *p == 'e' || *p == 'E')) {
      fail("floating-point or exponent not allowed in header integers");
      return v;
    }
    v.int_value = neg ? -n : n;
    return v;
  }

  void parse_string(std::string* out) {
    ++p;  // opening quote
    out->clear();
    while (p < end) {
      const char c = *p;
      if (c == '"') {
        ++p;
        return;
      }
      if (c == '\\') {
        ++p;
        if (p >= end) {
          fail("unterminated escape");
          return;
        }
        const char e = *p++;
        switch (e) {
          case '"': out->push_back('"'); break;
          case '\\': out->push_back('\\'); break;
          case '/': out->push_back('/'); break;
          case 'b': out->push_back('\b'); break;
          case 'f': out->push_back('\f'); break;
          case 'n': out->push_back('\n'); break;
          case 'r': out->push_back('\r'); break;
          case 't': out->push_back('\t'); break;
          case 'u': {
            if (end - p < 4) {
              fail("truncated \\u escape");
              return;
            }
            unsigned int cp = 0;
            for (int i = 0; i < 4; ++i) {
              const char h = *p++;
              const int d = (h >= '0' && h <= '9') ? (h - '0')
                             : (h >= 'a' && h <= 'f') ? (h - 'a' + 10)
                             : (h >= 'A' && h <= 'F') ? (h - 'A' + 10) : -1;
              if (d < 0) {
                fail("invalid \\u escape");
                return;
              }
              cp = (cp << 4) | static_cast<unsigned int>(d);
            }
            // Encode the BMP code point as UTF-8. \uXXXX yields a 16-bit value;
            // a lone surrogate (which tensor names will never contain) would
            // produce invalid UTF-8 bytes, which the manifest then carries as
            // an ordinary (unusable) name rather than a parse failure.
            if (cp < 0x80) {
              out->push_back(static_cast<char>(cp));
            } else if (cp < 0x800) {
              out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
              out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else {
              out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
              out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
              out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            break;
          }
          default:
            fail("invalid escape character");
            return;
        }
        continue;
      }
      if (static_cast<unsigned char>(c) < 0x20) {
        fail("unescaped control character in string");
        return;
      }
      out->push_back(c);
      ++p;
    }
    fail("unterminated string");
  }
};

// --- header validation ------------------------------------------------------

bool valid_dtype(const std::string& dtype) {
  static const char* const kDtypes[] = {"F64", "F32", "F16", "BF16", "I64",
                                        "I32", "I16", "I8", "U8",
                                        "F8_E4M3", "F8_E5M2"};
  for (const char* d : kDtypes) {
    if (dtype == d) return true;
  }
  return false;
}

}  // namespace

bool dtype_item_size(const std::string& dtype, int64_t* item_size) {
  struct Row {
    const char* name;
    int64_t size;
  };
  static const Row kRows[] = {
      {"F64", 8}, {"F32", 4}, {"F16", 2}, {"BF16", 2}, {"I64", 8},
      {"I32", 4}, {"I16", 2}, {"I8", 1}, {"U8", 1}, {"F8_E4M3", 1},
      {"F8_E5M2", 1},
  };
  for (const Row& r : kRows) {
    if (dtype == r.name) {
      *item_size = r.size;
      return true;
    }
  }
  return false;
}

// --- TensorManifest ---------------------------------------------------------

const TensorEntry* TensorManifest::find(const std::string& name) const {
  for (const auto& t : tensors) {
    if (t.name == name) return &t;
  }
  return nullptr;
}

int64_t TensorManifest::total_bytes() const {
  int64_t sum = 0;
  for (const auto& t : tensors) {
    if (t.nbytes < 0) return -1;
    int64_t next;
    if (!safe_add(sum, t.nbytes, &next)) return -1;
    sum = next;
  }
  return sum;
}

bool TensorManifest::verify() const {
  std::set<std::string> seen;
  for (const auto& t : tensors) {
    if (t.name.empty() || t.name == "__metadata__") return false;
    if (!seen.insert(t.name).second) return false;
    if (t.dtype.empty() || !valid_dtype(t.dtype)) return false;
    int64_t numel = 1;
    for (int64_t d : t.shape) {
      if (d < 0) return false;
      int64_t next;
      if (!safe_mul(numel, d, &next)) return false;
      numel = next;
    }
    if (t.numel != numel) return false;
    if (t.offset_begin < 0 || t.offset_end < t.offset_begin) return false;
    if (t.offset_end > data_region_bytes) return false;
    int64_t item;
    if (!dtype_item_size(t.dtype, &item)) return false;
    int64_t expected;
    if (!safe_mul(numel, item, &expected)) return false;
    if (t.nbytes != (t.offset_end - t.offset_begin)) return false;
    if (t.nbytes != expected) return false;
  }
  return true;
}

std::string TensorManifest::describe_issues() const {
  std::string out;
  std::set<std::string> seen;
  for (const auto& t : tensors) {
    if (t.name.empty() || t.name == "__metadata__") {
      out += "tensor with invalid name; ";
      continue;
    }
    if (!seen.insert(t.name).second) {
      out += "duplicate tensor: " + t.name + "; ";
      continue;
    }
    if (!valid_dtype(t.dtype)) out += "invalid dtype for " + t.name + "; ";
    int64_t numel = 1;
    bool shape_bad = false;
    for (int64_t d : t.shape) {
      if (d < 0) {
        out += "negative dim in " + t.name + "; ";
        shape_bad = true;
      } else {
        int64_t next;
        if (!safe_mul(numel, d, &next)) shape_bad = true;
        else numel = next;
      }
    }
    if (shape_bad && out.find("negative dim in " + t.name) == std::string::npos)
      out += "shape overflow in " + t.name + "; ";
    if (t.numel != numel) out += "numel mismatch for " + t.name + "; ";
    if (t.offset_begin < 0 || t.offset_end < t.offset_begin) {
      out += "invalid offsets for " + t.name + "; ";
    } else if (t.offset_end > data_region_bytes) {
      out += "out-of-bounds range for " + t.name + "; ";
    }
  }
  return out.empty() ? std::string("ok") : out;
}

void TensorManifest::diff_expected_names(
    const std::vector<std::string>& expected, std::vector<std::string>* missing,
    std::vector<std::string>* unexpected) const {
  if (missing) missing->clear();
  if (unexpected) unexpected->clear();
  std::set<std::string> expected_set;
  std::set<std::string> manifest_set;
  for (const auto& n : expected) expected_set.insert(n);
  for (const auto& t : tensors) manifest_set.insert(t.name);

  if (missing) {
    std::set<std::string> seen;
    for (const auto& n : expected) {
      if (manifest_set.count(n) == 0 && seen.insert(n).second) missing->push_back(n);
    }
  }
  if (unexpected) {
    for (const auto& t : tensors) {
      if (expected_set.count(t.name) == 0) unexpected->push_back(t.name);
    }
  }
}

// --- top-level parse ---------------------------------------------------------

bool parse_safetensors(const char* data, std::size_t data_size,
                       std::size_t file_size, TensorManifest* out,
                       std::string* error) {
  auto fail = [&](const char* code, const std::string& msg) {
    if (error) *error = make_error(code, msg);
    return false;
  };
  if (data == nullptr || out == nullptr) {
    return fail("invalid-args", "data and out must be non-null");
  }
  if (file_size < data_size) {
    return fail("invalid-args", "file_size smaller than buffer size");
  }
  if (data_size < 8) {
    return fail("truncated", "buffer shorter than 8-byte header length prefix");
  }
  uint64_t n64 = 0;
  le64(data, &n64);
  if (n64 == 0) {
    return fail("invalid-header", "header length is zero");
  }
  if (n64 > kMaxHeaderBytes) {
    return fail("unreasonable", "header length exceeds 64 MiB bound");
  }
  const std::size_t header_len = static_cast<std::size_t>(n64);
  if (data_size < 8 + header_len) {
    return fail("truncated",
                "file ends before full header (need 8+" + std::to_string(header_len) +
                    " bytes)");
  }

  JsonParser parser;
  parser.p = data + 8;
  parser.end = data + 8 + header_len;
  JsonValue root = parser.parse_value();
  if (parser.error) {
    return fail("malformed-json", parser.err);
  }
  parser.skip_ws();
  if (parser.p != parser.end) {
    return fail("malformed-json", "trailing content after header JSON");
  }
  if (root.kind != JsonValue::Kind::Object) {
    return fail("invalid-json", "header must be a JSON object");
  }

  TensorManifest manifest;
  manifest.header_bytes = static_cast<int64_t>(header_len);
  manifest.data_region_bytes =
      static_cast<int64_t>(file_size) - 8 - static_cast<int64_t>(header_len);

  std::set<std::string> seen;
  for (const auto& kv : root.obj) {
    const std::string& key = kv.first;
    // Uniform duplicate-key rejection (covers both tensors and __metadata__).
    if (!seen.insert(key).second) {
      return fail("duplicate", "duplicate header key: " + key);
    }
    if (key == "__metadata__") {
      if (kv.second.kind != JsonValue::Kind::Object) {
        return fail("invalid-field", "__metadata__ must be an object");
      }
      manifest.has_metadata = true;
      continue;
    }
    const JsonValue& v = kv.second;
    if (v.kind != JsonValue::Kind::Object) {
      return fail("invalid-field", "tensor entry must be an object: " + key);
    }

    TensorEntry te;
    te.name = key;

    const JsonValue* dtype = v.get("dtype");
    if (!dtype) {
      return fail("missing-field", "missing dtype for " + key);
    }
    if (dtype->kind != JsonValue::Kind::String) {
      return fail("invalid-dtype", "dtype must be a string for " + key);
    }
    te.dtype = dtype->str;
    if (!valid_dtype(te.dtype)) {
      return fail("invalid-dtype", "unknown dtype for " + key + ": " + te.dtype);
    }
    int64_t item = 0;
    if (!dtype_item_size(te.dtype, &item)) {
      return fail("invalid-dtype", "cannot map dtype to item size: " + key);
    }

    const JsonValue* shape = v.get("shape");
    if (!shape) {
      return fail("missing-field", "missing shape for " + key);
    }
    if (shape->kind != JsonValue::Kind::Array) {
      return fail("invalid-shape", "shape must be an array for " + key);
    }
    int64_t numel = 1;
    for (const auto& d : shape->arr) {
      if (d.kind != JsonValue::Kind::Int) {
        return fail("invalid-shape", "shape dim must be an integer for " + key);
      }
      if (d.int_value < 0) {
        return fail("invalid-shape", "negative dim in " + key);
      }
      int64_t next;
      if (!safe_mul(numel, d.int_value, &next)) {
        return fail("overflow", "shape product overflows int64 for " + key);
      }
      numel = next;
      te.shape.push_back(d.int_value);
    }
    te.numel = numel;

    const JsonValue* offsets = v.get("data_offsets");
    if (!offsets) {
      return fail("missing-field", "missing data_offsets for " + key);
    }
    if (offsets->kind != JsonValue::Kind::Array || offsets->arr.size() != 2) {
      return fail("invalid-offsets", "data_offsets must be [start, end] for " + key);
    }
    if (offsets->arr[0].kind != JsonValue::Kind::Int ||
        offsets->arr[1].kind != JsonValue::Kind::Int) {
      return fail("invalid-offsets", "data_offsets must be integers for " + key);
    }
    te.offset_begin = offsets->arr[0].int_value;
    te.offset_end = offsets->arr[1].int_value;
    if (te.offset_begin < 0 || te.offset_end < te.offset_begin) {
      return fail("invalid-offsets",
                  "offsets must be non-negative and ascending for " + key);
    }
    if (te.offset_end > manifest.data_region_bytes) {
      return fail("out-of-bounds", "data range exceeds file size for " + key);
    }
    te.nbytes = te.offset_end - te.offset_begin;

    int64_t expected = 0;
    if (!safe_mul(numel, item, &expected)) {
      return fail("overflow", "byte range overflows int64 for " + key);
    }
    if (te.nbytes != expected) {
      return fail("mismatch",
                  "byte range " + std::to_string(te.nbytes) +
                      " != numel*item_size " + std::to_string(expected) + " for " + key);
    }

    for (const auto& f : v.obj) {
      if (f.first != "dtype" && f.first != "shape" && f.first != "data_offsets") {
        return fail("unknown-field", "unknown tensor field: " + f.first);
      }
    }

    manifest.tensors.push_back(std::move(te));
  }

  *out = manifest;
  return true;
}

bool parse_safetensors_file(const std::string& path, TensorManifest* out,
                            std::string* error) {
  auto fail = [&](const char* code, const std::string& msg) {
    if (error) *error = make_error(code, msg);
    return false;
  };
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    return fail("io", "cannot open " + path);
  }
  const std::streamsize total = f.tellg();
  if (total < 0) {
    return fail("io", "cannot determine size of " + path);
  }
  const std::size_t file_size = static_cast<std::size_t>(total);
  if (file_size < 8) {
    return fail("truncated", "file shorter than 8-byte header prefix");
  }
  // Read only the bounded metadata prefix: 8-byte length + N header bytes.
  // The (potentially huge) data region is never allocated or read.
  f.seekg(0, std::ios::beg);
  char prefix[8];
  f.read(prefix, 8);
  if (f.gcount() != 8) {
    return fail("truncated", "short read of 8-byte prefix");
  }
  uint64_t n64 = 0;
  le64(prefix, &n64);
  if (n64 == 0) {
    return fail("invalid-header", "header length is zero");
  }
  if (n64 > kMaxHeaderBytes) {
    return fail("unreasonable", "header length exceeds 64 MiB bound");
  }
  const std::size_t header_len = static_cast<std::size_t>(n64);
  if (file_size < 8 + header_len) {
    return fail("truncated", "file ends before full header");
  }
  std::string buf(8 + header_len, '\0');
  std::memcpy(buf.data(), prefix, 8);
  f.seekg(8, std::ios::beg);
  f.read(buf.data() + 8, static_cast<std::streamsize>(header_len));
  if (f.gcount() != static_cast<std::streamsize>(header_len)) {
    return fail("truncated", "short read of header bytes");
  }
  return parse_safetensors(buf.data(), buf.size(), file_size, out, error);
}

}  // namespace granite4

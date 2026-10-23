// Minimal bounded JSON parser for config.json and similar flat model
// metadata. Header-only, stdlib-only, no execution.
//
// Supported: objects, arrays, strings (with escapes), numbers (double and
// int64), booleans, null. Depth-bounded. On success the root value is in
// *out; on failure *error receives a bounded diagnostic and false is
// returned. The parser rejects trailing content after the root value.
#ifndef GRANITE4_JSON_H
#define GRANITE4_JSON_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace granite4 {
namespace json {

struct Value {
  enum class Kind { Null, Bool, Int, Double, String, Array, Object };
  Kind kind = Kind::Null;
  bool bool_value = false;
  int64_t int_value = 0;
  double double_value = 0.0;
  std::string str;
  std::vector<Value> arr;
  std::vector<std::pair<std::string, Value>> obj;  // preserves order

  const Value* find(const std::string& key) const {
    if (kind != Kind::Object) return nullptr;
    for (const auto& kv : obj)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }

  // Convenience: integer field of an object; returns default when the key
  // is absent or not an integer/double.
  int64_t get_int(const std::string& key, int64_t def) const {
    const Value* v = find(key);
    if (v == nullptr) return def;
    if (v->kind == Kind::Int) return v->int_value;
    if (v->kind == Kind::Double) return static_cast<int64_t>(v->double_value);
    return def;
  }
  double get_double(const std::string& key, double def) const {
    const Value* v = find(key);
    if (v == nullptr) return def;
    if (v->kind == Kind::Double) return v->double_value;
    if (v->kind == Kind::Int) return static_cast<double>(v->int_value);
    return def;
  }
  bool get_bool(const std::string& key, bool def) const {
    const Value* v = find(key);
    if (v == nullptr) return def;
    if (v->kind == Kind::Bool) return v->bool_value;
    return def;
  }
  std::string get_string(const std::string& key) const {
    const Value* v = find(key);
    if (v == nullptr || v->kind != Kind::String) return "";
    return v->str;
  }
};

// Parse `text` (size bytes) into *out. Returns false + *error on failure.
bool parse(const char* text, std::size_t size, Value* out, std::string* error);

}  // namespace json
}  // namespace granite4

#endif  // GRANITE4_JSON_H

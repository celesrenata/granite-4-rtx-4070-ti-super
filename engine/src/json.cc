// Minimal bounded JSON parser implementation (see json.h).
//
// Derived from the safetensors header parser in safetensors.cc (same
// bounds, same error style) with two extensions required by model
// config files: floating-point numbers, and a depth/syntax bound large
// enough for config.json while still rejecting pathological nesting.

#include "json.h"

#include <cctype>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace granite4 {
namespace json {
namespace {

constexpr int kMaxJsonDepth = 64;

struct Parser {
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

  Value parse_value() {
    skip_ws();
    if (p >= end) {
      fail("unexpected end of JSON input");
      return {};
    }
    char c = *p;
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') {
      Value v;
      v.kind = Value::Kind::String;
      parse_string(&v.str);
      return v;
    }
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
    if (at("true")) {
      p += 4;
      Value v;
      v.kind = Value::Kind::Bool;
      v.bool_value = true;
      return v;
    }
    if (at("false")) {
      p += 5;
      Value v;
      v.kind = Value::Kind::Bool;
      v.bool_value = false;
      return v;
    }
    if (at("null")) {
      p += 4;
      Value v;
      v.kind = Value::Kind::Null;
      return v;
    }
    fail("expected a JSON value");
    return {};
  }

  Value parse_object() {
    Value v;
    v.kind = Value::Kind::Object;
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
      Value key;
      key.kind = Value::Kind::String;
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
      Value val = parse_value();
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

  Value parse_array() {
    Value v;
    v.kind = Value::Kind::Array;
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

  void parse_string(std::string* out) {
    ++p;  // opening quote
    while (p < end) {
      const char c = *p++;
      if (c == '"') {
        return;
      }
      if (c == '\\') {
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
    }
    fail("unterminated string");
  }

  Value parse_number() {
    const char* start = p;
    if (p < end && *p == '-') ++p;
    bool is_float = false;
    while (p < end && std::isdigit(static_cast<unsigned char>(*p)) != 0) ++p;
    if (p < end && *p == '.') {
      is_float = true;
      ++p;
      while (p < end && std::isdigit(static_cast<unsigned char>(*p)) != 0) ++p;
    }
    if (p < end && (*p == 'e' || *p == 'E')) {
      is_float = true;
      ++p;
      if (p < end && (*p == '+' || *p == '-')) ++p;
      while (p < end && std::isdigit(static_cast<unsigned char>(*p)) != 0) ++p;
    }
    const char* digits_start = (*start == '-') ? start + 1 : start;
    const std::size_t len = static_cast<std::size_t>(p - start);
    if (len == 0 || (len == 1 && *start == '-')) {
      fail("malformed number");
      return {};
    }
    Value v;
    if (is_float) {
      const std::string num(start, len);
      try {
        v.double_value = std::stod(num);
      } catch (...) {
        fail("number out of range");
        return {};
      }
      v.kind = Value::Kind::Double;
      if (std::isnan(v.double_value) || std::isinf(v.double_value)) {
        fail("non-finite number");
        return {};
      }
    } else {
      // Strict int64 parse with overflow guard (strtoll, no whitespace).
      const std::string num(start, len);
      const std::size_t max_digits = (*start == '-') ? 19 : 18;
      if (num.size() - (num[0] == '-' ? 1 : 0) > max_digits) {
        try {
          v.double_value = std::stod(num);
          v.kind = Value::Kind::Double;
        } catch (...) {
          fail("number out of range");
          return {};
        }
      } else {
        try {
          v.int_value = std::stoll(num);
          v.kind = Value::Kind::Int;
        } catch (...) {
          try {
            v.double_value = std::stod(num);
            v.kind = Value::Kind::Double;
          } catch (...) {
            fail("number out of range");
            return {};
          }
        }
      }
    }
    (void)digits_start;
    return v;
  }
};

}  // namespace

bool parse(const char* text, std::size_t size, Value* out, std::string* error) {
  if (text == nullptr || out == nullptr || size == 0) {
    if (error != nullptr) *error = "empty input";
    return false;
  }
  Parser ps;
  ps.p = text;
  ps.end = text + size;
  Value root = ps.parse_value();
  if (ps.error) {
    if (error != nullptr) *error = ps.err;
    return false;
  }
  ps.skip_ws();
  if (ps.p != ps.end) {
    if (error != nullptr) *error = "trailing content after JSON value";
    return false;
  }
  *out = std::move(root);
  return true;
}

}  // namespace json
}  // namespace granite4

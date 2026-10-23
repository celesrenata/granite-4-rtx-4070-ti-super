#include "json.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <string>

int main() {
  granite4::json::Value root;
  std::string error;
  const char* text = R"({"layers":40,"eps":1e-5,"scale":0.0078125,"enabled":true,"zero":0})";
  assert(granite4::json::parse(text, std::strlen(text), &root, &error));
  assert(root.kind == granite4::json::Value::Kind::Object);
  assert(root.get_int("layers", -1) == 40);
  assert(root.get_int("zero", -1) == 0);
  assert(std::abs(root.get_double("eps", 0.0) - 1e-5) < 1e-12);
  assert(std::abs(root.get_double("scale", 0.0) - 0.0078125) < 1e-12);
  assert(root.get_bool("enabled", false));

  assert(!granite4::json::parse("{\"x\":}", 6, &root, &error));
  assert(!granite4::json::parse("{\"x\":1} trailing", 17, &root, &error));
  return 0;
}

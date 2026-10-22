// Host-only tests for the Phase-1 safetensors metadata foundation.
//
// Covers: a valid synthetic header (in-memory and via file), truncation,
// header-length bounds, malformed JSON, duplicate / unknown / missing tensor
// fields, invalid dtype / shape / offset forms, overflow, and out-of-bounds
// data ranges. Also exercises the immutable manifest API (verify,
// describe_issues, diff_expected_names, total_bytes) and the narrow C ABI
// validator.
//
// No CUDA required; run via ctest after a host-only build.
#include "capi.h"
#include "safetensors.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void expect(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

// Build a complete in-memory safetensors buffer:
// 8-byte LE header length + header JSON + `data_bytes` zero bytes.
std::string make_buffer(const std::string& header, std::size_t data_bytes) {
  std::string buf(8, '\0');
  const uint64_t n = static_cast<uint64_t>(header.size());
  for (int i = 0; i < 8; ++i) {
    buf[i] = static_cast<char>((n >> (8 * i)) & 0xFF);
  }
  buf += header;
  buf += std::string(data_bytes, '\0');
  return buf;
}

const char* kValidHeader =
    R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}})";

bool try_parse(const std::string& buf, granite4::TensorManifest* out,
               std::string* error) {
  return granite4::parse_safetensors(buf.data(), buf.size(), buf.size(), out,
                                     error);
}

// Expect parse to succeed.
void expect_ok(const char* what, const std::string& buf) {
  granite4::TensorManifest m;
  std::string err;
  if (!try_parse(buf, &m, &err)) {
    std::printf("FAIL: %s (got error: %s)\n", what, err.c_str());
    ++g_failures;
  } else if (!m.verify()) {
    std::printf("FAIL: %s (verify: %s)\n", what, m.describe_issues().c_str());
    ++g_failures;
  }
}

// Expect parse to fail with the given error code prefix ("code: ...").
void expect_err(const char* what, const std::string& buf, const char* code) {
  granite4::TensorManifest m;
  std::string err;
  if (try_parse(buf, &m, &err)) {
    std::printf("FAIL: %s (expected failure, parse succeeded)\n", what);
    ++g_failures;
    return;
  }
  const std::string prefix = std::string(code) + ":";
  if (err.rfind(prefix, 0) != 0) {
    std::printf("FAIL: %s (error '%s', expected prefix '%s')\n", what, err.c_str(),
                prefix.c_str());
    ++g_failures;
  }
  if (err.size() > 255) {
    std::printf("FAIL: %s (diagnostic longer than 255 bytes)\n", what);
    ++g_failures;
  }
}

}  // namespace

int main() {
  using namespace granite4;

  // --- valid synthetic header, in memory -----------------------------------
  {
    const std::string buf = make_buffer(kValidHeader, /*data_bytes=*/12);
    TensorManifest m;
    std::string err;
    expect(try_parse(buf, &m, &err), "valid header parses");
    expect(m.tensors.size() == 1, "one tensor");
    if (m.tensors.size() == 1) {
      const TensorEntry& t = m.tensors[0];
      expect(t.name == "w", "tensor name");
      expect(t.dtype == "BF16", "tensor dtype");
      expect(t.shape.size() == 2 && t.shape[0] == 2 && t.shape[1] == 3,
             "tensor shape");
      expect(t.offset_begin == 0 && t.offset_end == 12, "tensor offsets");
      expect(t.numel == 6, "numel");
      expect(t.nbytes == 12, "nbytes");
    }
    expect(m.data_region_bytes == 12, "data region size");
    expect(m.header_bytes == static_cast<int64_t>(std::strlen(kValidHeader)),
           "header byte count");
    expect(m.has_metadata == false, "no __metadata__ by default");
    expect(m.verify(), "valid manifest verifies");
    expect(m.describe_issues() == "ok", "describe_issues reports ok");
    expect(m.total_bytes() == 12, "total_bytes");
    expect(m.find("w") != nullptr, "find(w)");
    expect(m.find("other") == nullptr, "find(other) is null");

    // Name delta (callers supply the names; the module never invents any).
    std::vector<std::string> missing, unexpected;
    m.diff_expected_names({"w", "extra"}, &missing, &unexpected);
    expect(missing.size() == 1 && missing[0] == "extra", "missing name");
    expect(unexpected.empty(), "no unexpected names");
    m.diff_expected_names({"other"}, &missing, &unexpected);
    expect(missing.size() == 1 && missing[0] == "other", "missing name 2");
    expect(unexpected.size() == 1 && unexpected[0] == "w", "unexpected name");
    m.diff_expected_names({"w"}, &missing, &unexpected);
    expect(missing.empty() && unexpected.empty(), "exact match delta");
  }

  // --- __metadata__ accepted and ignored ------------------------------------
  {
    const std::string buf = make_buffer(
        R"({"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})",
        16);
    TensorManifest m;
    std::string err;
    expect(try_parse(buf, &m, &err) && m.verify(), "metadata + tensor parses");
    expect(m.has_metadata, "has_metadata flag");
    expect(m.tensors.size() == 1 && m.tensors[0].name == "w",
           "__metadata__ is not a tensor");
  }

  // --- empty object header is structurally valid ----------------------------
  {
    expect_ok("empty header parses", make_buffer("{}", 0));
  }

  // --- truncation ------------------------------------------------------------
  {
    // Buffer shorter than the 8-byte length prefix.
    expect_err("buffer < 8 bytes", "1234567", "truncated");
    // Declared header longer than what follows.
    const std::string buf = make_buffer(kValidHeader, 0);
    std::string short_buf = buf.substr(0, 8 + std::strlen(kValidHeader) - 3);
    expect_err("truncated header", short_buf, "truncated");
    // Declared header length of 0.
    std::string zero(8, '\0');
    expect_err("zero header length", zero, "invalid-header");
  }

  // --- unreasonable header length (> 64 MiB), no allocation -------------------
  {
    std::string buf(8, '\0');
    const uint64_t n = kMaxHeaderBytes + 1;
    for (int i = 0; i < 8; ++i) buf[i] = static_cast<char>((n >> (8 * i)) & 0xFF);
    expect_err("header > 64 MiB rejected", buf, "unreasonable");
  }

  // --- malformed JSON ----------------------------------------------------------
  {
    expect_err("unterminated object", make_buffer("{", 0), "malformed-json");
    expect_err("missing value", make_buffer(R"({"w":})", 0), "malformed-json");
    expect_err("missing colon", make_buffer(R"({"w" "BF16"})", 0),
               "malformed-json");
    expect_err("trailing garbage",
               make_buffer(std::string(kValidHeader) + "x", 0),
               "malformed-json");
    // JSON permits whitespace around the root value; it is accepted.
    expect_ok("trailing whitespace accepted",
              make_buffer(std::string(kValidHeader) + " \n", 12));
    expect_err("bare array root", make_buffer("[1]", 0), "invalid-json");
    expect_err("unterminated string",
               make_buffer(std::string("{\"w\":\"x"), 0), "malformed-json");
    expect_err("unescaped control char",
               make_buffer(std::string("{\"w\"\t"), 0), "malformed-json");
    expect_err("bad escape",
               make_buffer(std::string("{\"w\":\"a\\q\"}"), 0),
               "malformed-json");
    // Floats are rejected where the header requires strict integers.
    expect_err("float dim",
               make_buffer(
                   R"({"w":{"dtype":"BF16","shape":[1.5],"data_offsets":[0,2]}})",
                   2),
               "malformed-json");
    expect_err("float offset",
               make_buffer(
                   R"({"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2.0]}})",
                   2),
               "malformed-json");
  }

  // --- duplicate / unknown / missing fields ------------------------------------
  {
    const std::string dup =
        std::string(kValidHeader).substr(0, std::strlen(kValidHeader) - 1) +
        ",\"w\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[0,12]}}";
    expect_err("duplicate tensor key", make_buffer(dup, 12), "duplicate");
    expect_err("unknown tensor field",
               make_buffer(
                   R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12],"quant":"fp8"}})",
                   12),
               "unknown-field");
    expect_err("missing dtype",
               make_buffer(
                   R"({"w":{"shape":[2,3],"data_offsets":[0,12]}})", 12),
               "missing-field");
    expect_err("missing shape",
               make_buffer(
                   R"({"w":{"dtype":"BF16","data_offsets":[0,12]}})", 12),
               "missing-field");
    expect_err("missing data_offsets",
               make_buffer(R"({"w":{"dtype":"BF16","shape":[2,3]}})", 12),
               "missing-field");
    expect_err("non-object tensor",
               make_buffer(R"({"w":"BF16"})", 0), "invalid-field");
    expect_err("__metadata__ non-object",
               make_buffer(R"({"__metadata__":5})", 0), "invalid-field");
    expect_err("duplicate __metadata__",
               make_buffer(R"({"__metadata__":{},"__metadata__":{}})", 0),
               "duplicate");
  }

  // --- invalid dtype / shape / offset forms --------------------------------------
  {
    expect_err("unknown dtype",
               make_buffer(
                   R"({"w":{"dtype":"FP8","shape":[2,3],"data_offsets":[0,12]}})",
                   12),
               "invalid-dtype");
    expect_err("dtype not a string",
               make_buffer(
                   R"({"w":{"dtype":7,"shape":[2,3],"data_offsets":[0,12]}})", 12),
               "invalid-dtype");
    expect_err("shape not an array",
               make_buffer(
                   R"({"w":{"dtype":"BF16","shape":6,"data_offsets":[0,12]}})", 12),
               "invalid-shape");
    expect_err("shape dim not an integer",
               make_buffer(
                   R"({"w":{"dtype":"BF16","shape":["2"],"data_offsets":[0,2]}})",
                   2),
               "invalid-shape");
    expect_err("negative dim",
               make_buffer(
                   R"({"w":{"dtype":"BF16","shape":[-1],"data_offsets":[0,2]}})",
                   2),
               "invalid-shape");
    expect_err("offsets not a 2-element array",
               make_buffer(
                   R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0]}})",
                   12),
               "invalid-offsets");
    expect_err("offsets not integers",
               make_buffer(
                   R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,"x"]}})",
                   12),
               "invalid-offsets");
    expect_err("descending offsets",
               make_buffer(
                   R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[12,0]}})",
                   12),
               "invalid-offsets");
    expect_err("negative offset",
               make_buffer(
                   R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[-4,12]}})",
                   12),
               "invalid-offsets");
    expect_err("byte range / numel mismatch",
               make_buffer(
                   R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,10]}})",
                   10),
               "mismatch");
  }

  // --- overflow and out-of-bounds ---------------------------------------------------
  {
    expect_err("shape product overflow",
               make_buffer(
                   R"({"w":{"dtype":"BF16","shape":[9223372036854775807,2],"data_offsets":[0,0]}})",
                   0),
               "overflow");
    expect_err("offset beyond file end",
               make_buffer(
                   R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,13]}})",
                   12),
               "out-of-bounds");
    // Two tensors whose ranges fit are fine; a second one starting mid-data is
    // legal as long as ranges are consistent (overlaps are not Phase-1
    // invariants, so overlap alone must not fail).
    const std::string buf = make_buffer(
        R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})",
        12);
    expect_ok("two tensors, non-overlapping fit", buf);
  }

  // --- file-based parse ---------------------------------------------------------------
  {
    const char* tmp_path = "granite4_safetensors_test.tmp.safetensors";
    {
      const std::string buf = make_buffer(kValidHeader, 12);
      std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
      f.write(buf.data(), static_cast<std::streamsize>(buf.size()));
      expect(static_cast<bool>(f), "temp file written");
    }

    TensorManifest m;
    std::string err;
    expect(parse_safetensors_file(tmp_path, &m, &err) && m.verify(),
           "file parse valid");
    expect(m.tensors.size() == 1, "file parse tensor count");

    std::string e2;
    TensorManifest m2;
    expect(!parse_safetensors_file("granite4_safetensors_does_not_exist.safetensors",
                                   &m2, &e2),
           "missing file fails");
    expect(e2.rfind("io:", 0) == 0, "missing file error code");

    // Small file claiming a > 64 MiB header.
    {
      std::string big(8, '\0');
      const uint64_t n = kMaxHeaderBytes + 1;
      for (int i = 0; i < 8; ++i)
        big[i] = static_cast<char>((n >> (8 * i)) & 0xFF);
      {
        std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
        f.write(big.data(), 8);
      }
      std::string e3;
      TensorManifest m3;
      expect(!parse_safetensors_file(tmp_path, &m3, &e3) &&
                 e3.rfind("unreasonable:", 0) == 0,
             "file with oversized header claim rejected");
    }

    // File-based truncation: the 8-byte prefix declares the full header
    // length, but the file ends before the header bytes are present.
    {
      std::string cut(8, '\0');
      for (int i = 0; i < 8; ++i)
        cut[i] = static_cast<char>(
            (static_cast<uint64_t>(std::strlen(kValidHeader)) >> (8 * i)) &
            0xFF);
      cut += std::string(kValidHeader).substr(0, 4);  // only 4 bytes present
      {
        std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
        f.write(cut.data(), static_cast<std::streamsize>(cut.size()));
      }
      std::string e4;
      TensorManifest m4;
      expect(!parse_safetensors_file(tmp_path, &m4, &e4) &&
                 e4.rfind("truncated:", 0) == 0,
             "file truncated before full header");
    }

    std::remove(tmp_path);
  }

  // --- manifest invariants: duplicate detection --------------------------------------
  {
    // The parser rejects duplicates at parse time; verify() also guards the
    // invariant for manifests assembled elsewhere.
    TensorManifest m;
    m.data_region_bytes = 8;
    for (const char* name : {"a", "a"}) {
      TensorEntry t;
      t.name = name;
      t.dtype = "F32";
      t.shape = {2};
      t.offset_begin = 0;
      t.offset_end = 8;
      t.numel = 2;
      t.nbytes = 8;
      m.tensors.push_back(t);
    }
    expect(!m.verify(), "verify catches duplicate names");
    const std::string issues = m.describe_issues();
    expect(issues.find("duplicate") != std::string::npos,
           "describe_issues mentions duplicate");
  }

  // --- C ABI ---------------------------------------------------------------------------
  {
    const char* tmp_path = "granite4_safetensors_abi.tmp.safetensors";
    {
      const std::string buf = make_buffer(kValidHeader, 12);
      std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
      f.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    }
    granite4_safetensors_result r{};
    granite4_validate_safetensors(tmp_path, &r);
    expect(r.ok == 1, "C ABI valid file ok");
    expect(r.tensor_count == 1, "C ABI tensor count");
    expect(r.diagnostic[0] == '\0', "C ABI empty diagnostic on success");

    granite4_safetensors_result r2{};
    granite4_validate_safetensors("granite4_safetensors_nope.safetensors", &r2);
    expect(r2.ok == 0, "C ABI missing file fails");
    expect(std::strlen(r2.diagnostic) > 0, "C ABI diagnostic present on failure");

    granite4_safetensors_result r3{};
    granite4_validate_safetensors(nullptr, &r3);
    expect(r3.ok == 0, "C ABI null path fails cleanly");

    granite4_safetensors_result r4{};
    granite4_validate_safetensors(tmp_path, &r4);
    expect(r4.ok == 1, "C ABI repeat call");
    std::remove(tmp_path);
  }

  if (g_failures == 0) {
    std::printf("PASS: all engine safetensors checks\n");
    return 0;
  }
  std::printf("%d check(s) failed\n", g_failures);
  return 1;
}

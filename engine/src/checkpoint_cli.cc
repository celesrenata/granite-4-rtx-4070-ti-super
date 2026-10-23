// granite4-checkpoint CLI (Phase 1, host-only).
//
//   granite4-checkpoint verify <checkpoint_dir>
//       Streams a full verification (config + all shards + checksums) and
//       writes the JSON report to stdout. Exit 0 iff the report is ok.
//
//   granite4-checkpoint export <checkpoint_dir> <out.safetensors> [--per-channel]
//       Runs verification, then exports the quantized FP8 e4m3 checkpoint.
//       Exit 0 iff verification is ok and the export wrote.
//
// No CUDA. Nothing is loaded into a device; "verify" streams the checkpoint
// in bounded chunks and reports per-shard / per-tensor SHA-256.
#include "checkpoint.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int print_usage(const char* prog) {
  std::fprintf(stderr,
               "usage:\n"
               "  %s verify <checkpoint_dir>\n"
               "  %s export <checkpoint_dir> <out.safetensors> [--per-channel]\n",
               prog, prog);
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return print_usage(argv[0]);
  const std::string cmd = argv[1];
  if (cmd == "verify") {
    if (argc != 3) return print_usage(argv[0]);
    const char* dir = argv[2];
    // Probe with a null buffer to learn the required size (returns 0 by
    // design and sets `needed`), then write the report into a sized buffer.
    long needed = 0;
    granite4_verify_checkpoint(dir, nullptr, 0, &needed);
    if (needed <= 0) {
      std::fprintf(stderr, "granite4-checkpoint: probe failed\n");
      return 1;
    }
    std::vector<char> buf(static_cast<std::size_t>(needed) + 1);
    long used = 0;
    const int rc = granite4_verify_checkpoint(dir, buf.data(),
                                               static_cast<long>(buf.size()), &used);
    if (rc != 1) {
      std::fprintf(stderr, "granite4-checkpoint: report write failed\n");
      return 1;
    }
    std::fwrite(buf.data(), 1, static_cast<std::size_t>(used), stdout);
    std::fputc('\n', stdout);
    // Exit 0 only when the report says ok:true.
    const std::string report(buf.data(), static_cast<std::size_t>(used));
    const bool ok = report.find("\"ok\":true") != std::string::npos &&
                    report.find("\"config_ok\":true") != std::string::npos;
    return ok ? 0 : 1;
  }
  if (cmd == "export") {
    bool per_channel = false;
    const char* dir = nullptr;
    const char* out_path = nullptr;
    for (int i = 2; i < argc; ++i) {
      if (std::strcmp(argv[i], "--per-channel") == 0) per_channel = true;
      else if (dir == nullptr) dir = argv[i];
      else if (out_path == nullptr) out_path = argv[i];
      else return print_usage(argv[0]);
    }
    if (dir == nullptr || out_path == nullptr) return print_usage(argv[0]);
    granite4::CheckpointReport report;
    std::string error;
    if (!granite4::verify_checkpoint(dir, &report, &error)) {
      std::fprintf(stderr, "granite4-checkpoint: verification failed: %s\n", error.c_str());
      return 1;
    }
    std::string exp_err;
    if (!granite4::export_fp8_checkpoint(dir, &report, out_path, per_channel, &exp_err)) {
      std::fprintf(stderr, "granite4-checkpoint: export failed: %s\n", exp_err.c_str());
      return 1;
    }
    std::printf("exported %s: %zu tensors, %ld bytes, ok=1\n", out_path,
                report.tensor_digests.size(),
                static_cast<long>(report.total_bytes));
    return 0;
  }
  return print_usage(argv[0]);
}

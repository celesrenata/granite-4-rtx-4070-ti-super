// SHA-256 (FIPS 180-4). Stdlib-only, incremental, no platform crypto.
//
// hex() is non-mutating: it works on a shallow copy of the running state
// (all fields are trivially copyable), so a digest may be taken at any point
// without destroying the in-progress hash.
#include "checkpoint.h"

#include <cstring>

namespace granite4 {

namespace {

// Round constants: first 32 bits of the fractional parts of the cube roots
// of the first 64 primes.
const uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

// One compression block over explicit state (used by hex() on a copy).
void transform_block(uint32_t st[8], const uint8_t block[64]) {
  uint32_t w[64];
  for (int t = 0; t < 16; ++t) {
    w[t] = (static_cast<uint32_t>(block[4 * t]) << 24) |
           (static_cast<uint32_t>(block[4 * t + 1]) << 16) |
           (static_cast<uint32_t>(block[4 * t + 2]) << 8) |
           static_cast<uint32_t>(block[4 * t + 3]);
  }
  for (int t = 16; t < 64; ++t) {
    // Sigma0 = ROTR7 ^ ROTR18 ^ SHR3 (FIPS 180-4; a plain 3-bit shift, NOT a
    // rotation - the 1-bit rotation variant loses the top bit).
    const uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
    const uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
    w[t] = w[t - 16] + s0 + w[t - 7] + s1;
  }

  uint32_t a = st[0], b = st[1], c = st[2], d = st[3];
  uint32_t e = st[4], f = st[5], g = st[6], hh = st[7];

  for (int t = 0; t < 64; ++t) {
    const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const uint32_t ch = (e & f) ^ (~e & g);
    const uint32_t t1 = hh + s1 + ch + kK[t] + w[t];
    const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + maj;
    hh = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  st[0] += a;
  st[1] += b;
  st[2] += c;
  st[3] += d;
  st[4] += e;
  st[5] += f;
  st[6] += g;
  st[7] += hh;
}

}  // namespace

Sha256::Sha256() {
  // First 32 bits of the fractional parts of the square roots of the first
  // 8 primes (FIPS 180-4, SHA-256 initial hash values).
  h[0] = 0x6a09e667;
  h[1] = 0xbb67ae85;
  h[2] = 0x3c6ef372;
  h[3] = 0xa54ff53a;
  h[4] = 0x510e527f;
  h[5] = 0x9b05688c;
  h[6] = 0x1f83d9ab;
  h[7] = 0x5be0cd19;
}

void Sha256::transform(const uint8_t block[64]) { transform_block(h, block); }

void Sha256::update(const void* data, std::size_t nbytes) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  total_bytes += nbytes;
  if (buflen > 0) {
    const std::size_t take = 64 - buflen;
    const std::size_t n = nbytes < take ? nbytes : take;
    std::memcpy(buf + buflen, p, n);
    buflen += n;
    p += n;
    nbytes -= n;
    if (buflen == 64) {
      transform(buf);
      buflen = 0;
    }
  }
  while (nbytes >= 64) {
    transform(p);
    p += 64;
    nbytes -= 64;
  }
  if (nbytes > 0) {
    std::memcpy(buf, p, nbytes);
    buflen = nbytes;
  }
}

std::string Sha256::hex() const {
  // Canonical padding over a copy: the running state is never mutated.
  const uint64_t bits = total_bytes * 8;
  uint32_t st[8];
  std::memcpy(st, h, sizeof st);
  uint8_t buffered[64];
  std::memcpy(buffered, buf, 64);

  auto finalize_block = [&](uint8_t block[64]) {
    std::memset(block, 0, 64);
    std::memcpy(block, buffered, buflen);
    block[buflen] = 0x80;
    if (buflen + 9 > 64) {
      // 0x80 alone fills the first block; the length gets its own block.
      transform_block(st, block);
      uint8_t len_block[64];
      std::memset(len_block, 0, sizeof len_block);
      for (int i = 0; i < 8; ++i)
        len_block[56 + i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
      transform_block(st, len_block);
    } else {
      for (int i = 0; i < 8; ++i)
        block[56 + i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
      transform_block(st, block);
    }
  };
  uint8_t tail[64];
  finalize_block(tail);

  static const char kHex[] = "0123456789abcdef";
  std::string out(64, '0');
  // 8 hex chars per state word (4 bytes, big-endian digest).
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) {
      const int byte = (st[i] >> (24 - 8 * j)) & 0xFF;
      const int pos = 8 * i + 2 * j;
      out[pos] = kHex[byte >> 4];
      out[pos + 1] = kHex[byte & 0xF];
    }
  }
  return out;
}

}  // namespace granite4

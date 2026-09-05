// Minimal host-side PNG writer. No zlib dependency: PNG's zlib stream is emitted using
// deflate *stored* (uncompressed) blocks, which are legal and trivially encoded. Files are
// larger than a compressed PNG but directly viewable, which is the point.
#pragma once
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

namespace g4gpu::vis {

inline uint32_t crc32_of(const uint8_t* data, size_t n, uint32_t crc = 0xFFFFFFFFu) {
  static uint32_t table[256];
  static bool ready = false;
  if (!ready) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) { c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1); }
      table[i] = c;
    }
    ready = true;
  }
  for (size_t i = 0; i < n; ++i) { crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8); }
  return crc;
}

inline uint32_t adler32_of(const uint8_t* data, size_t n) {
  uint32_t a = 1, b = 0;
  for (size_t i = 0; i < n; ++i) {
    a = (a + data[i]) % 65521u;
    b = (b + a) % 65521u;
  }
  return (b << 16) | a;
}

inline void push_be32(std::vector<uint8_t>& v, uint32_t x) {
  v.push_back(static_cast<uint8_t>(x >> 24));
  v.push_back(static_cast<uint8_t>(x >> 16));
  v.push_back(static_cast<uint8_t>(x >> 8));
  v.push_back(static_cast<uint8_t>(x));
}

inline void push_chunk(std::vector<uint8_t>& out, const char type[5],
                       const std::vector<uint8_t>& data) {
  push_be32(out, static_cast<uint32_t>(data.size()));
  std::vector<uint8_t> tc;
  tc.insert(tc.end(), type, type + 4);
  tc.insert(tc.end(), data.begin(), data.end());
  out.insert(out.end(), tc.begin(), tc.end());
  push_be32(out, crc32_of(tc.data(), tc.size()) ^ 0xFFFFFFFFu);
}

/// Writes a top-down 24-bit RGB image (3 bytes per pixel, row 0 at the top).
inline bool write_png_rgb(const char* path, const uint8_t* rgb, int width, int height) {
  // Raw zlib payload: each scanline is a filter byte (0 = none) followed by the RGB row.
  std::vector<uint8_t> raw;
  raw.reserve(static_cast<size_t>(height) * (1 + 3 * width));
  for (int y = 0; y < height; ++y) {
    raw.push_back(0);
    const uint8_t* row = rgb + static_cast<size_t>(y) * width * 3;
    raw.insert(raw.end(), row, row + static_cast<size_t>(width) * 3);
  }

  // zlib header, then stored deflate blocks of at most 65535 bytes, then adler32.
  std::vector<uint8_t> z;
  z.push_back(0x78);
  z.push_back(0x01);
  size_t off = 0;
  while (off < raw.size()) {
    const size_t chunk = (raw.size() - off < 65535u) ? (raw.size() - off) : 65535u;
    const bool last = (off + chunk >= raw.size());
    z.push_back(last ? 1 : 0);  // BFINAL, BTYPE = 00 (stored)
    z.push_back(static_cast<uint8_t>(chunk & 0xFF));
    z.push_back(static_cast<uint8_t>((chunk >> 8) & 0xFF));
    const uint16_t nlen = static_cast<uint16_t>(~chunk);
    z.push_back(static_cast<uint8_t>(nlen & 0xFF));
    z.push_back(static_cast<uint8_t>((nlen >> 8) & 0xFF));
    z.insert(z.end(), raw.begin() + off, raw.begin() + off + chunk);
    off += chunk;
  }
  push_be32(z, adler32_of(raw.data(), raw.size()));

  std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  std::vector<uint8_t> ihdr;
  push_be32(ihdr, static_cast<uint32_t>(width));
  push_be32(ihdr, static_cast<uint32_t>(height));
  ihdr.push_back(8);  // bit depth
  ihdr.push_back(2);  // colour type 2 = truecolour RGB
  ihdr.push_back(0);  // deflate
  ihdr.push_back(0);  // adaptive filtering
  ihdr.push_back(0);  // no interlace
  push_chunk(png, "IHDR", ihdr);
  push_chunk(png, "IDAT", z);
  push_chunk(png, "IEND", {});

  FILE* f = std::fopen(path, "wb");
  if (f == nullptr) { return false; }
  std::fwrite(png.data(), 1, png.size(), f);
  std::fclose(f);
  return true;
}

}  // namespace g4gpu::vis

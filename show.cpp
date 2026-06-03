#include <cstdint>

#include <xsimd/xsimd.hpp>

bool FindHashBlockXsimd(const std::uint32_t* block, const std::uint32_t* salt, std::uint32_t key) {
  using batch = xsimd::batch<uint32_t>;
  const batch mask =
      batch(uint32_t{1})
      << xsimd::bitwise_rshift<27>(batch(key) * batch::load_unaligned(salt));
  const batch miss = xsimd::bitwise_andnot(mask, batch::load_unaligned(block));
  // `miss != 0` (one extra vpcmpeqd) is deliberate: reinterpreting `miss`
  // directly as a batch_bool would skip the compare but feed non-canonical
  // lane values into batch_bool, which relies on xsimd's AVX2 backend
  // lowering none() to a whole-register vptest. That lowering is not part
  // of xsimd's documented contract.
  return xsimd::none(miss != batch(uint32_t{0}));
}

bool FindHashBlockScalar(const std::uint32_t* block, const std::uint32_t* salt, std::uint32_t key) {
  constexpr int kBitsSetPerBlock = 8;
  uint32_t miss = 0;
  for (int i = 0; i < kBitsSetPerBlock; ++i) {
    const uint32_t mask = static_cast<uint32_t>(1) << ((key * salt[i]) >> 27);
    miss |= (~block[i] & mask);
  }
  return miss == 0;
}

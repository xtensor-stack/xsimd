/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <random>
#include <numeric>

#include "test_utils.hpp"

static_assert(xsimd::default_arch::supported(), "default arch must be supported");
static_assert(xsimd::supported_architectures::contains<xsimd::default_arch>(), "default arch is supported");
static_assert(xsimd::all_architectures::contains<xsimd::default_arch>(), "default arch is a valid arch");
//static_assert(!(xsimd::x86_arch::supported() && xsimd::arm::supported()), "either x86 or arm, but not both");

struct check_supported {
  template<class Arch>
  void operator()(Arch) const {
    static_assert(Arch::supported(), "not supported?");
  }
};

TEST(arch, supported)
{
  xsimd::supported_architectures::for_each(check_supported{});
}

struct check_available {
  template<class Arch>
  void operator()(Arch) const {
    EXPECT_TRUE(Arch::available());
  }
};

TEST(arch, available)
{
  EXPECT_TRUE(xsimd::default_arch::available());
}

struct sum {
  template<class Arch, class T>
  T operator()(Arch, T const* data, unsigned size)
  {
    using batch = xsimd::batch<T, Arch>;
    batch acc(static_cast<T>(0));
    const unsigned n = size / batch::size * batch::size;
    for(unsigned i = 0; i != n; i += batch::size)
        acc += batch::load_unaligned(data + i);
    T star_acc = xsimd::hadd(acc);
    for(unsigned i = n; i < size; ++i)
      star_acc += data[i];
    return star_acc;
  }
};

TEST(arch, dispatcher)
{
  float data[17] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f };
  float ref = std::accumulate(std::begin(data), std::end(data), 0.f);

  // platform specific
  {
    auto dispatched = xsimd::dispatch(sum{});
    float res = dispatched(data, 17);
    EXPECT_EQ(ref, res);
  }

#if XSIMD_WITH_AVX && XSIMD_WITH_SSE2
  static_assert(xsimd::supported_architectures::contains<xsimd::avx>() && xsimd::supported_architectures::contains<xsimd::sse2>(), "consistent supported architectures");
  {
    auto dispatched = xsimd::dispatch<sum, xsimd::arch_list<xsimd::avx, xsimd::sse2>>(sum{});
    float res = dispatched(data, 17);
    EXPECT_EQ(ref, res);
  }
#endif
}

#ifdef XSIMD_ENABLE_FALLBACK
// FIXME: this should be named scalar
TEST(arch, scalar)
{
  float data[17] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
  float ref = std::accumulate(std::begin(data), std::end(data), 0);

  float res = sum{}(xsimd::arch::scalar{}, data, 17);
  EXPECT_EQ(ref, res);

}
#endif

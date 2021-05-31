#ifndef XSIMD_GENERIC_ARCH_HPP
#define XSIMD_GENERIC_ARCH_HPP

namespace xsimd {

  struct generic {
    static constexpr bool supported() { return true; }
    static constexpr bool available() { return true; }
    protected:
    static constexpr unsigned version(unsigned major, unsigned minor, unsigned patch) { return major * 100u + minor * 10u + patch; }
  };

}
#endif


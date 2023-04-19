// compile with -msse2
#include "sum.hpp"
template float sum::operator()<xsimd::sse2, float>(xsimd::sse2, float const*, unsigned);

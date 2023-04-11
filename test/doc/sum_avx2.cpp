// compile with -mavx2
#include "sum.hpp"
template float sum::operator()<xsimd::avx2, float>(xsimd::avx2, float const*, unsigned);

#include "xsimd/xsimd.hpp"
#include <cstddef>
#include <vector>

void mean(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& res)
{
    using b_type = xsimd::batch<double, xsimd::avx>;
    std::size_t inc = b_type::size;
    std::size_t size = res.size();
    // size for which the vectorization is possible
    std::size_t vec_size = size - size % inc;
    for (std::size_t i = 0; i < vec_size; i += inc)
    {
        b_type avec = b_type::load_unaligned(&a[i]);
        b_type bvec = b_type::load_unaligned(&b[i]);
        b_type rvec = (avec + bvec) / 2;
        rvec.store_unaligned(&res[i]);
    }
    // Remaining part that cannot be vectorize
    for (std::size_t i = vec_size; i < size; ++i)
    {
        res[i] = (a[i] + b[i]) / 2;
    }
}

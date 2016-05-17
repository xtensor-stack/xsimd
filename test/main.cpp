#include <iostream>
#include <fstream>
#include <map>
#include "nxsimd/config/nx_simd_include.hpp"
#include "nxsimd/memory/nx_aligned_allocator.hpp"
#include "nxsimd/types/nx_sse_double.hpp"
#include "nxsimd/types/nx_sse_float.hpp"
#include "nxsimd/types/nx_avx_double.hpp"
#include "nxsimd/types/nx_avx_float.hpp"
#include "nx_simd_common_test.hpp"

namespace nxsimd
{
    template <class V, size_t N, size_t A>
    bool test_simd(std::ostream& out, const std::string& name)
    {
        simd_basic_tester<V, N, A> tester(name);
        return test_simd_common(out, tester);
    }

}

bool test_sse_float_basic()
{
    std::ofstream out("log/sse_float_basic.log", std::ios_base::out);
    bool res = nxsimd::test_simd<nxsimd::vector4f, 4, 16>(out, "sse float");
    return res;
}

bool test_sse_double_basic()
{
    std::ofstream out("log/sse_double_basic.log", std::ios_base::out);
    bool res = nxsimd::test_simd<nxsimd::vector2d, 2, 16>(out, "sse double");
    return res;
}

bool test_avx_float_basic()
{
    std::ofstream out("log/avx_float_basic.log", std::ios_base::out);
    bool res = nxsimd::test_simd<nxsimd::vector8f, 8, 32>(out, "avx float");
    return res;
}

bool test_avx_double_basic()
{
    std::ofstream out("log/avx_double_basic.log", std::ios_base::out);
    bool res = nxsimd::test_simd<nxsimd::vector4d, 4, 32>(out, "avx double");
    return res;
}


int main(int argc, char* argv[])
{
    using test_list_type = std::map<std::string, bool (*)()>;
    test_list_type test_list;
    test_list["sse float basic"] = test_sse_float_basic;
    test_list["sse double basic"] = test_sse_double_basic;
    test_list["avx float basic"] = test_avx_float_basic;
    test_list["avx double basic"] = test_avx_double_basic;

    int nb_failed = 0;
    for(auto iter : test_list)
    {
        std::cout << iter.first << " : ";
        bool res = (iter.second)();
        std::cout << (res ? "OK" : "FAILED") << std::endl;
        if(!res) ++nb_failed;
    }

    std::cout << std::endl;

    int nb_passed = test_list.size() - nb_failed;
    std::cout << "Nb tests passed : " <<  nb_passed << std::endl;
    std::cout << "Nb tests failed : " <<  nb_failed << std::endl;

    return 0;
}


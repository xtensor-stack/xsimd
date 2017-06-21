/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xsimd_benchmark.hpp"
#include <map>

void benchmark_operation()
{
    //std::size_t size = 9984;
    std::size_t size = 20000;
    xsimd::run_benchmark_2op(xsimd::add_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_2op(xsimd::sub_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_2op(xsimd::mul_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_2op(xsimd::div_fn(), std::cout, size, 1000);
}

void benchmark_exp_log()
{
    std::size_t size = 20000;
    xsimd::run_benchmark_1op(xsimd::exp_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_1op(xsimd::exp2_fn(), std::cout, size, 100);
    xsimd::run_benchmark_1op(xsimd::expm1_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_1op(xsimd::log_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_1op(xsimd::log2_fn(), std::cout, size, 100);
    xsimd::run_benchmark_1op(xsimd::log10_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_1op(xsimd::log1p_fn(), std::cout, size, 1000);
}

void benchmark_trigo()
{
    std::size_t size = 20000;
    xsimd::run_benchmark_1op(xsimd::sin_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_1op(xsimd::cos_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_1op(xsimd::tan_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_1op(xsimd::asin_fn(), std::cout, size, 1000, xsimd::init_method::arctrigo);
    xsimd::run_benchmark_1op(xsimd::acos_fn(), std::cout, size, 1000, xsimd::init_method::arctrigo);
    xsimd::run_benchmark_1op(xsimd::atan_fn(), std::cout, size, 1000, xsimd::init_method::arctrigo);
}

void benchmark_hyperbolic()
{
    std::size_t size = 20000;
    xsimd::run_benchmark_1op(xsimd::sinh_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_1op(xsimd::cosh_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_1op(xsimd::tanh_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_1op(xsimd::asinh_fn(), std::cout, size, 100);
    xsimd::run_benchmark_1op(xsimd::acosh_fn(), std::cout, size, 100);
    xsimd::run_benchmark_1op(xsimd::atanh_fn(), std::cout, size, 100);
}

void benchmark_power()
{
    std::size_t size = 20000;
    xsimd::run_benchmark_2op(xsimd::pow_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_1op(xsimd::sqrt_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_1op(xsimd::cbrt_fn(), std::cout, size, 100);
    xsimd::run_benchmark_2op(xsimd::hypot_fn(), std::cout, size, 1000);
}

void benchmark_rounding()
{
    std::size_t size = 20000;
    xsimd::run_benchmark_1op(xsimd::ceil_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_1op(xsimd::floor_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_1op(xsimd::trunc_fn(), std::cout, size, 1000);
    xsimd::run_benchmark_1op(xsimd::round_fn(), std::cout, size, 100);
    xsimd::run_benchmark_1op(xsimd::nearbyint_fn(), std::cout, size, 100);
    xsimd::run_benchmark_1op(xsimd::rint_fn(), std::cout, size, 100);
}

int main(int argc, char* argv[])
{
    if (argc > 1)
    {
        std::map<std::string, void(*)()> fn_map;
        fn_map["op"] = benchmark_operation;
        fn_map["exp"] = benchmark_exp_log;
        fn_map["trigo"] = benchmark_trigo;
        fn_map["hyperbolic"] = benchmark_hyperbolic;
        fn_map["power"] = benchmark_power;
        fn_map["rounding"] = benchmark_rounding;

        if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")
        {
            std::cout << "Avalaible options:" << std::endl;
            std::cout << "op        : run benchmark on aithmetic operations" << std::endl;
            std::cout << "exp       : run benchmark on exponential and logarithm functions" << std::endl;
            std::cout << "trigo     : run benchmark on trigonomeric functions" << std::endl;
            std::cout << "hyperbolic: run benchmark on hyperbolic functions" << std::endl;
            std::cout << "power     : run benchmark on power functions" << std::endl;
            std::cout << "rounding  : run benchmark on rounding functions" << std::endl;
        }
        else
        {
            for (int i = 1; i < argc; ++i)
            {
                fn_map[std::string(argv[i])]();
            }
        }
    }
    else
    {
        benchmark_operation();
        benchmark_exp_log();
        benchmark_trigo();
        benchmark_hyperbolic();
        benchmark_power();
        benchmark_rounding();
    }
    return 0;
}

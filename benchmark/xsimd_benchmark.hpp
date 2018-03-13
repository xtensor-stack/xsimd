/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_BENCHMARK_HPP
#define XSIMD_BENCHMARK_HPP

#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include "xsimd/xsimd.hpp"

namespace xsimd
{
    template <class T>
    std::string batch_name();

    template <> inline std::string batch_name<batch<float, 4>>() { return "sse/neon float"; }
    template <> inline std::string batch_name<batch<double, 2>>() { return "sse/neon double"; }
    template <> inline std::string batch_name<batch<float, 8>>() { return "avx float"; }
    template <> inline std::string batch_name<batch<double, 4>>() { return "avx double"; }

    using duration_type = std::chrono::duration<double, std::milli>;

    template <class T>
    using bench_vector = std::vector<T, xsimd::aligned_allocator<T, XSIMD_DEFAULT_ALIGNMENT>>;

    template <class T>
    void init_benchmark(bench_vector<T>& lhs, bench_vector<T>& rhs, bench_vector<T>& res, size_t size)
    {
        lhs.resize(size);
        rhs.resize(size);
        res.resize(size);
        for (size_t i = 0; i < size; ++i)
        {
            lhs[i] = T(0.5) + std::sqrt(T(i)) * T(9.) / T(size);
            rhs[i] = T(10.2) / T(i + 2) + T(0.25);
        }
    }

    template <class T>
    void init_benchmark_arctrigo(bench_vector<T>& lhs, bench_vector<T>& rhs, bench_vector<T>& res, size_t size)
    {
        lhs.resize(size);
        rhs.resize(size);
        res.resize(size);
        for (size_t i = 0; i < size; ++i)
        {
            lhs[i] = T(-1.) + T(2.) * T(i) / T(size);
            rhs[i] = T(i) / T(i + 2) + T(0.25);
        }
    }

    enum class init_method
    {
        classic,
        arctrigo
    };

    template <class F, class V>
    duration_type benchmark_scalar(F f, V& lhs, V& res, std::size_t number)
    {
        size_t s = lhs.size();
        duration_type t_res = duration_type::max();
        for (std::size_t count = 0; count < number; ++count)
        {
            auto start = std::chrono::steady_clock::now();
            for (size_t i = 0; i < s; ++i)
            {
                res[i] = f(lhs[i]);
            }
            auto end = std::chrono::steady_clock::now();
            auto tmp = end - start;
            t_res = tmp < t_res ? tmp : t_res;
        }
        return t_res;
    }

    template <class F, class V>
    duration_type benchmark_scalar(F f, V& lhs, V& rhs, V& res, std::size_t number)
    {
        size_t s = lhs.size();
        duration_type t_res = duration_type::max();
        for (std::size_t count = 0; count < number; ++count)
        {
            auto start = std::chrono::steady_clock::now();
            for (size_t i = 0; i < s; ++i)
            {
               res[i] = f(lhs[i], rhs[i]);
            }
            auto end = std::chrono::steady_clock::now();
            auto tmp = end - start;
            t_res = tmp < t_res ? tmp : t_res;
        }
        return t_res;
    }

    template <class B, class F, class V>
    duration_type benchmark_simd(F f, V& lhs, V& res, std::size_t number)
    {
        std::size_t s = lhs.size();
        duration_type t_res = duration_type::max();
        for (std::size_t count = 0; count < number; ++count)
        {
            auto start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < s; i += B::size)
            {
                B blhs(&lhs[i], aligned_mode());
                B bres = f(blhs);
                bres.store_aligned(&res[i]);
            }
            auto end = std::chrono::steady_clock::now();
            auto tmp = end - start;
            t_res = tmp < t_res ? tmp : t_res;
        }
        return t_res;
    }

    template <class B, class F, class V>
    duration_type benchmark_simd_unrolled(F f, V& lhs, V& res, std::size_t number)
    {
        std::size_t s = lhs.size();
        std::size_t inc = 4 * B::size;
        duration_type t_res = duration_type::max();
        for (std::size_t count = 0; count < number; ++count)
        {
            auto start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < s; i += inc)
            {
                size_t j = i + B::size;
                size_t k = j + B::size;
                size_t l = k + B::size;
                B blhs(&lhs[i], aligned_mode()), blhs2(&lhs[j], aligned_mode()),
                  blhs3(&lhs[k], aligned_mode()), blhs4(&lhs[l], aligned_mode());
                B bres = f(blhs);
                B bres2 = f(blhs2);
                B bres3 = f(blhs3);
                B bres4 = f(blhs4);
                bres.store_aligned(&res[i]);
                bres2.store_aligned(&res[j]);
                bres3.store_aligned(&res[k]);
                bres4.store_aligned(&res[l]);
            }
            auto end = std::chrono::steady_clock::now();
            auto tmp = end - start;
            t_res = tmp < t_res ? tmp : t_res;
        }
        return t_res;
    }

    template <class B, class F, class V>
    duration_type benchmark_simd(F f, V& lhs, V& rhs, V& res, std::size_t number)
    {
        std::size_t s = lhs.size();
        duration_type t_res = duration_type::max();
        for (std::size_t count = 0; count < number; ++count)
        {
            auto start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < s; i += B::size)
            {
                B blhs(&lhs[i], aligned_mode()), brhs(&rhs[i], aligned_mode());
                B bres = f(blhs, brhs);
                bres.store_aligned(&res[i]);
            }
            auto end = std::chrono::steady_clock::now();
            auto tmp = end - start;
            t_res = tmp < t_res ? tmp : t_res;
        }
        return t_res;
    }

    template <class B, class F, class V>
    duration_type benchmark_simd_unrolled(F f, V& lhs, V& rhs, V& res, std::size_t number)
    {
        std::size_t s = lhs.size();
        std::size_t inc = 4 * B::size;
        duration_type t_res = duration_type::max();
        for (std::size_t count = 0; count < number; ++count)
        {
            auto start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < s; i += inc)
            {
                size_t j = i + B::size;
                size_t k = j + B::size;
                size_t l = k + B::size;
                B blhs(&lhs[i], aligned_mode()), brhs(&rhs[i], aligned_mode()),
                  blhs2(&lhs[j], aligned_mode()), brhs2(&rhs[j], aligned_mode());
                B blhs3(&lhs[k], aligned_mode()), brhs3(&rhs[k], aligned_mode()),
                  blhs4(&lhs[l], aligned_mode()), brhs4(&rhs[l], aligned_mode());
                B bres = f(blhs, brhs);
                B bres2 = f(blhs2, brhs2);
                B bres3 = f(blhs3, brhs3);
                B bres4 = f(blhs4, brhs4);
                bres.store_aligned(&res[i]);
                bres2.store_aligned(&res[j]);
                bres3.store_aligned(&res[k]);
                bres4.store_aligned(&res[l]);
            }
            auto end = std::chrono::steady_clock::now();
            auto tmp = end - start;
            t_res = tmp < t_res ? tmp : t_res;
        }
        return t_res;
    }

    template <class F, class OS>
    void run_benchmark_1op(F f, OS& out, std::size_t size, std::size_t iter, init_method init = init_method::classic)
    {
        bench_vector<float> f_lhs, f_rhs, f_res;
        bench_vector<double> d_lhs, d_rhs, d_res;

        switch (init)
        {
        case init_method::classic:
            init_benchmark(f_lhs, f_rhs, f_res, size);
            init_benchmark(d_lhs, d_rhs, d_res, size);
            break;
        case init_method::arctrigo:
            init_benchmark_arctrigo(f_lhs, f_rhs, f_res, size);
            init_benchmark_arctrigo(d_lhs, d_rhs, d_res, size);
            break;
        default:
            init_benchmark(f_lhs, f_rhs, f_res, size);
            init_benchmark(d_lhs, d_rhs, d_res, size);
            break;
        }

        duration_type t_float_scalar = benchmark_scalar(f, f_lhs, f_res, iter);
        duration_type t_double_scalar = benchmark_scalar(f, d_lhs, d_res, iter);

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
        duration_type t_float_sse = benchmark_simd<batch<float, 4>>(f, f_lhs, f_res, iter);
        duration_type t_float_sse_u = benchmark_simd_unrolled<batch<float, 4>>(f, f_lhs, f_res, iter);
        duration_type t_double_sse = benchmark_simd<batch<double, 2>>(f, d_lhs, d_res, iter);
        duration_type t_double_sse_u = benchmark_simd_unrolled<batch<double, 2>>(f, d_lhs, d_res, iter);
#endif
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
        duration_type t_float_avx = benchmark_simd<batch<float, 8>>(f, f_lhs, f_res, iter);
        duration_type t_float_avx_u = benchmark_simd_unrolled<batch<float, 8>>(f, f_lhs, f_res, iter);
        duration_type t_double_avx = benchmark_simd<batch<double, 4>>(f, d_lhs, d_res, iter);
        duration_type t_double_avx_u = benchmark_simd_unrolled<batch<double, 4>>(f, d_lhs, d_res, iter);
#endif
#if defined(XSIMD_ARM_INSTR_SET)
        duration_type t_float_neon = benchmark_simd<batch<float, 4>>(f, f_lhs, f_res, iter);
        duration_type t_float_neon_u = benchmark_simd_unrolled<batch<float, 4>>(f, f_lhs, f_res, iter);
        duration_type t_double_neon = benchmark_simd<batch<double, 2>>(f, d_lhs, d_res, iter);
        duration_type t_double_neon_u = benchmark_simd_unrolled<batch<double, 2>>(f, d_lhs, d_res, iter);
#endif

        out << "============================" << std::endl;
        out << f.name() << std::endl;
        out << "scalar float   : " << t_float_scalar.count() << "ms" << std::endl;
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
        out << "sse float      : " << t_float_sse.count() << "ms" << std::endl;
        out << "sse float unr  : " << t_float_sse_u.count() << "ms" << std::endl;
#endif
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
        out << "avx float      : " << t_float_avx.count() << "ms" << std::endl;
        out << "avx float unr  : " << t_float_avx_u.count() << "ms" << std::endl;
#endif
#if defined(XSIMD_ARM_INSTR_SET)
        out << "neon float     : " << t_float_neon.count() << "ms" << std::endl;
        out << "neon float unr : " << t_float_neon_u.count() << "ms" << std::endl;
#endif
        out << "scalar double  : " << t_double_scalar.count() << "ms" << std::endl;
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
        out << "sse double     : " << t_double_sse.count() << "ms" << std::endl;
        out << "sse double unr : " << t_double_sse_u.count() << "ms" << std::endl;
#endif
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
        out << "avx double     : " << t_double_avx.count() << "ms" << std::endl;
        out << "avx double unr : " << t_double_avx_u.count() << "ms" << std::endl;
#endif
#if defined(XSIMD_ARM_INSTR_SET)
        out << "neon double    : " << t_double_neon.count() << "ms" << std::endl;
        out << "neon double unr: " << t_double_neon_u.count() << "ms" << std::endl;
#endif
        out << "============================" << std::endl;
    }

    template <class F, class OS>
    void run_benchmark_2op(F f, OS& out, std::size_t size, std::size_t iter)
    {
        bench_vector<float> f_lhs, f_rhs, f_res;
        bench_vector<double> d_lhs, d_rhs, d_res;

        init_benchmark(f_lhs, f_rhs, f_res, size);
        init_benchmark(d_lhs, d_rhs, d_res, size);

        duration_type t_float_scalar = benchmark_scalar(f, f_lhs, f_rhs, f_res, iter);
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
        duration_type t_float_sse = benchmark_simd<batch<float, 4>>(f, f_lhs, f_rhs, f_res, iter);
        duration_type t_float_sse_u = benchmark_simd_unrolled<batch<float, 4>>(f, f_lhs, f_rhs, f_res, iter);
#endif
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
        duration_type t_float_avx = benchmark_simd<batch<float, 8>>(f, f_lhs, f_rhs, f_res, iter);
        duration_type t_float_avx_u = benchmark_simd_unrolled<batch<float, 8>>(f, f_lhs, f_rhs, f_res, iter);
#endif
        duration_type t_double_scalar = benchmark_scalar(f, d_lhs, d_rhs, d_res, iter);
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
        duration_type t_double_sse = benchmark_simd<batch<double, 2>>(f, d_lhs, d_rhs, d_res, iter);
        duration_type t_double_sse_u = benchmark_simd_unrolled<batch<double, 2>>(f, d_lhs, d_rhs, d_res, iter);
#endif
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
        duration_type t_double_avx = benchmark_simd<batch<double, 4>>(f, d_lhs, d_rhs, d_res, iter);
        duration_type t_double_avx_u = benchmark_simd_unrolled<batch<double, 4>>(f, d_lhs, d_rhs, d_res, iter);
#endif
#if defined(XSIMD_ARM_INSTR_SET)
        duration_type t_float_neon = benchmark_simd<batch<float, 4>>(f, f_lhs, f_rhs, f_res, iter);
        duration_type t_float_neon_u = benchmark_simd_unrolled<batch<float, 4>>(f, f_lhs, f_rhs, f_res, iter);
        duration_type t_double_neon = benchmark_simd<batch<double, 2>>(f, d_lhs, d_rhs, d_res, iter);
        duration_type t_double_neon_u = benchmark_simd_unrolled<batch<double, 2>>(f, d_lhs, d_rhs, d_res, iter);
#endif

        out << "============================" << std::endl;
        out << f.name() << std::endl;
        out << "scalar float   : " << t_float_scalar.count() << "ms" << std::endl;
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
        out << "sse float      : " << t_float_sse.count() << "ms" << std::endl;
        out << "sse float unr  : " << t_float_sse_u.count() << "ms" << std::endl;
#endif
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
        out << "avx float      : " << t_float_avx.count() << "ms" << std::endl;
        out << "avx float unr  : " << t_float_avx_u.count() << "ms" << std::endl;
#endif
#if defined(XSIMD_ARM_INSTR_SET)
        out << "neon float     : " << t_float_neon.count() << "ms" << std::endl;
        out << "neon float unr : " << t_float_neon_u.count() << "ms" << std::endl;
#endif
        out << "scalar double  : " << t_double_scalar.count() << "ms" << std::endl;
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
        out << "sse double     : " << t_double_sse.count() << "ms" << std::endl;
        out << "sse double unr : " << t_double_sse_u.count() << "ms" << std::endl;
#endif
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
        out << "avx double     : " << t_double_avx.count() << "ms" << std::endl;
        out << "avx double unr : " << t_double_avx_u.count() << "ms" << std::endl;
#endif
#if defined(XSIMD_ARM_INSTR_SET)
        out << "neon double    : " << t_double_neon.count() << "ms" << std::endl;
        out << "neon double unr: " << t_double_neon_u.count() << "ms" << std::endl;
#endif
        out << "============================" << std::endl;
    }

#define DEFINE_OP_FUNCTOR_2OP(OP, NAME)\
    struct NAME##_fn {\
        template <class T>\
        inline T operator()(const T& lhs, const T& rhs) const { return lhs OP rhs; }\
        inline std::string name() const { return #NAME; }\
    }

#define DEFINE_FUNCTOR_1OP(FN)\
    struct FN##_fn {\
        template <class T>\
        inline T operator()(const T& x) const { using std::FN; using xsimd::FN; return FN(x); }\
        inline std::string name() const { return #FN; }\
    }

#define DEFINE_FUNCTOR_2OP(FN)\
    struct FN##_fn{\
        template <class T>\
        inline T operator()(const T&lhs, const T& rhs) const { using std::FN; using xsimd::FN; return FN(lhs, rhs); }\
        inline std::string name() const { return #FN; }\
    }

DEFINE_OP_FUNCTOR_2OP(+, add);
DEFINE_OP_FUNCTOR_2OP(-, sub);
DEFINE_OP_FUNCTOR_2OP(*, mul);
DEFINE_OP_FUNCTOR_2OP(/, div);

DEFINE_FUNCTOR_1OP(exp);
DEFINE_FUNCTOR_1OP(exp2);
DEFINE_FUNCTOR_1OP(expm1);
DEFINE_FUNCTOR_1OP(log);
DEFINE_FUNCTOR_1OP(log10);
DEFINE_FUNCTOR_1OP(log2);
DEFINE_FUNCTOR_1OP(log1p);

DEFINE_FUNCTOR_1OP(sin);
DEFINE_FUNCTOR_1OP(cos);
DEFINE_FUNCTOR_1OP(tan);
DEFINE_FUNCTOR_1OP(asin);
DEFINE_FUNCTOR_1OP(acos);
DEFINE_FUNCTOR_1OP(atan);

DEFINE_FUNCTOR_1OP(sinh);
DEFINE_FUNCTOR_1OP(cosh);
DEFINE_FUNCTOR_1OP(tanh);
DEFINE_FUNCTOR_1OP(asinh);
DEFINE_FUNCTOR_1OP(acosh);
DEFINE_FUNCTOR_1OP(atanh);

DEFINE_FUNCTOR_2OP(pow);
DEFINE_FUNCTOR_1OP(sqrt);
DEFINE_FUNCTOR_1OP(cbrt);
DEFINE_FUNCTOR_2OP(hypot);

DEFINE_FUNCTOR_1OP(ceil);
DEFINE_FUNCTOR_1OP(floor);
DEFINE_FUNCTOR_1OP(trunc);
DEFINE_FUNCTOR_1OP(round);
DEFINE_FUNCTOR_1OP(nearbyint);
DEFINE_FUNCTOR_1OP(rint);

}

#endif

/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_API_TEST_HPP
#define XSIMD_API_TEST_HPP

#include "xsimd_basic_test.hpp"
#include "xsimd_complex_basic_test.hpp"
#include "xsimd/xsimd.hpp"

namespace xsimd
{

    /*************
     * load test *
     *************/

    template <class T>
    inline bool test_simd_api_load(std::ostream& out, T& tester)
    {
        using int32_batch = typename T::int32_batch;
        using int64_batch = typename T::int64_batch;
        using float_batch = typename T::float_batch;
        using double_batch = typename T::double_batch;
        using int32_vector = typename T::int32_vector;
        using int64_vector = typename T::int64_vector;
        using float_vector = typename T::float_vector;
        using double_vector = typename T::double_vector;
        using char_vector = typename T::char_vector;
        using uchar_vector = typename T::uchar_vector;

        int32_batch i32bres;
        int64_batch i64bres;
        float_batch fbres;
        double_batch dbres;

        int32_vector i32vres(float_batch::size);
        int64_vector i64vres(float_batch::size);
        float_vector fvres(float_batch::size);
        double_vector dvres(float_batch::size);
        char_vector cvres(float_batch::size);
        uchar_vector ucvres(float_batch::size);

        int32_vector i32vres2(double_batch::size);
        int64_vector i64vres2(double_batch::size);
        float_vector fvres2(double_batch::size);
        double_vector dvres2(double_batch::size);
        char_vector cvres2(float_batch::size);
        uchar_vector ucvres2(float_batch::size);

        bool success = true;
        bool tmp_success = true;

        std::string name = tester.name;
        std::string name_shift = std::string(name.size(), '-');
        std::string dash(8, '-');
        std::string space(8, ' ');

        out << dash << name_shift << dash << std::endl;
        out << space << name << space << std::endl;
        out << dash << name_shift << dash << std::endl;

        // float

        std::string topic = "load float   -> float  : ";
        fbres = load_simd<float>(tester.f_vec.data(), aligned_mode());
        store_simd(fvres.data(), fbres, aligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "loadu float  -> float  : ";
        fbres = load_simd<float>(tester.f_vec.data(), unaligned_mode());
        store_simd(fvres.data(), fbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "load int32   -> float  : ";
        fbres = load_simd<int32_t, float>(tester.i32_vec.data(), aligned_mode());
        store_simd(fvres.data(), fbres, aligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;
        
        topic = "loadu int32  -> float  : ";
        fbres = load_simd<int32_t, float>(tester.i32_vec.data(), unaligned_mode());
        store_simd(fvres.data(), fbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "load int64   -> float  : ";
        fbres = load_simd<int64_t, float>(tester.i64_vec.data(), aligned_mode());
        store_simd(fvres.data(), fbres, aligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "loadu int64  -> float  : ";
        fbres = load_simd<int64_t, float>(tester.i64_vec.data(), unaligned_mode());
        store_simd(fvres.data(), fbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "load double  -> float  : ";
        fbres = load_simd<double, float>(tester.d_vec.data(), aligned_mode());
        store_simd(fvres.data(), fbres, aligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "loadu double -> float  : ";
        fbres = load_simd<double, float>(tester.d_vec.data(), unaligned_mode());
        store_simd(fvres.data(), fbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "load char    -> float  : ";
        fbres = load_simd<char, float>(tester.c_vec.data(), aligned_mode());
        store_simd(fvres.data(), fbres, aligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "loadu char   -> float  : ";
        fbres = load_simd<char, float>(tester.c_vec.data(), unaligned_mode());
        store_simd(fvres.data(), fbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "load uchar   -> float  : ";
        fbres = load_simd<unsigned char, float>(tester.uc_vec.data(), aligned_mode());
        store_simd(fvres.data(), fbres, aligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "loadu uchar  -> float  : ";
        fbres = load_simd<unsigned char, float>(tester.uc_vec.data(), unaligned_mode());
        store_simd(fvres.data(), fbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        // double

        topic = "load float   -> double : ";
        dbres = load_simd<float, double>(tester.f_vec.data(), aligned_mode());
        store_simd(dvres2.data(), dbres, aligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "loadu float  -> double : ";
        dbres = load_simd<float, double>(tester.f_vec.data(), unaligned_mode());
        store_simd(dvres2.data(), dbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "load int32   -> double : ";
        dbres = load_simd<int32_t, double>(tester.i32_vec.data(), aligned_mode());
        store_simd(dvres2.data(), dbres, aligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "loadu int32  -> double : ";
        dbres = load_simd<int32_t, double>(tester.i32_vec.data(), unaligned_mode());
        store_simd(dvres2.data(), dbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "load int64   -> double : ";
        dbres = load_simd<int64_t, double>(tester.i64_vec.data(), aligned_mode());
        store_simd(dvres2.data(), dbres, aligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "loadu int64  -> double : ";
        dbres = load_simd<int64_t, double>(tester.i64_vec.data(), unaligned_mode());
        store_simd(dvres2.data(), dbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "load double  -> double : ";
        dbres = load_simd<double>(tester.d_vec2.data(), aligned_mode());
        store_simd(dvres2.data(), dbres, aligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "loadu double -> double : ";
        dbres = load_simd<double>(tester.d_vec2.data(), unaligned_mode());
        store_simd(dvres2.data(), dbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "load char    -> double : ";
        dbres = load_simd<char, double>(tester.c_vec.data(), aligned_mode());
        store_simd(dvres2.data(), dbres, aligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "loadu char   -> double : ";
        dbres = load_simd<char, double>(tester.c_vec.data(), unaligned_mode());
        store_simd(dvres2.data(), dbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "load uchar   -> double : ";
        dbres = load_simd<unsigned char, double>(tester.uc_vec.data(), aligned_mode());
        store_simd(dvres2.data(), dbres, aligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "loadu uchar  -> double : ";
        dbres = load_simd<unsigned char, double>(tester.uc_vec.data(), unaligned_mode());
        store_simd(dvres2.data(), dbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        // int32

        topic = "load float   -> int32  : ";
        i32bres = load_simd<float, int32_t>(tester.f_vec.data(), aligned_mode());
        store_simd(i32vres.data(), i32bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "loadu float  -> int32  : ";
        i32bres = load_simd<float, int32_t>(tester.f_vec.data(), unaligned_mode());
        store_simd(i32vres.data(), i32bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "load int32   -> int32  : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), aligned_mode());
        store_simd(i32vres.data(), i32bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "loadu int32  -> int32  : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), unaligned_mode());
        store_simd(i32vres.data(), i32bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "load int64   -> int32  : ";
        i32bres = load_simd<int64_t, int32_t>(tester.i64_vec.data(), aligned_mode());
        store_simd(i32vres.data(), i32bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "loadu int64  -> int32  : ";
        i32bres = load_simd<int64_t, int32_t>(tester.i64_vec.data(), unaligned_mode());
        store_simd(i32vres.data(), i32bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "load double  -> int32  : ";
        i32bres = load_simd<double, int32_t>(tester.d_vec.data(), aligned_mode());
        store_simd(i32vres.data(), i32bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "loadu double -> int32  : ";
        i32bres = load_simd<double, int32_t>(tester.d_vec.data(), unaligned_mode());
        store_simd(i32vres.data(), i32bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "load char    -> int32  : ";
        i32bres = load_simd<char, int32_t>(tester.c_vec.data(), aligned_mode());
        store_simd(i32vres.data(), i32bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "loadu char   -> int32  : ";
        i32bres = load_simd<char, int32_t>(tester.c_vec.data(), unaligned_mode());
        store_simd(i32vres.data(), i32bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "load uchar   -> int32  : ";
        i32bres = load_simd<unsigned char, int32_t>(tester.uc_vec.data(), aligned_mode());
        store_simd(i32vres.data(), i32bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "loadu uchar  -> int32  : ";
        i32bres = load_simd<unsigned char, int32_t>(tester.uc_vec.data(), unaligned_mode());
        store_simd(i32vres.data(), i32bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        // int64

        topic = "load float   -> int64  : ";
        i64bres = load_simd<float, int64_t>(tester.f_vec.data(), aligned_mode());
        store_simd(i64vres2.data(), i64bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "loadu float  -> int64  : ";
        i64bres = load_simd<float, int64_t>(tester.f_vec.data(), unaligned_mode());
        store_simd(i64vres2.data(), i64bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "load int32   -> int64  : ";
        i64bres = load_simd<int32_t, int64_t>(tester.i32_vec.data(), aligned_mode());
        store_simd(i64vres2.data(), i64bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "loadu int32  -> int64  : ";
        i64bres = load_simd<int32_t, int64_t>(tester.i32_vec.data(), unaligned_mode());
        store_simd(i64vres2.data(), i64bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "load int64   -> int64  : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), aligned_mode());
        store_simd(i64vres2.data(), i64bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "loadu int64  -> int64  : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), unaligned_mode());
        store_simd(i64vres2.data(), i64bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "load double  -> int64  : ";
        i64bres = load_simd<double, int64_t>(tester.d_vec.data(), aligned_mode());
        store_simd(i64vres2.data(), i64bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "loadu double -> int64  : ";
        i64bres = load_simd<double, int64_t>(tester.d_vec.data(), unaligned_mode());
        store_simd(i64vres2.data(), i64bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "load char    -> int64  : ";
        i64bres = load_simd<char, int64_t>(tester.c_vec.data(), aligned_mode());
        store_simd(i64vres2.data(), i64bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "loadu char   -> int64  : ";
        i64bres = load_simd<char, int64_t>(tester.c_vec.data(), unaligned_mode());
        store_simd(i64vres2.data(), i64bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "load uchar   -> int64  : ";
        i64bres = load_simd<unsigned char, int64_t>(tester.uc_vec.data(), aligned_mode());
        store_simd(i64vres2.data(), i64bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "loadu uchar  -> int64  : ";
        i64bres = load_simd<unsigned char, int64_t>(tester.uc_vec.data(), unaligned_mode());
        store_simd(i64vres2.data(), i64bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        return success;
    }

    /**************
     * store test *
     **************/

    template <class T>
    inline bool test_simd_api_store(std::ostream& out, T& tester)
    {
        using int32_batch = typename T::int32_batch;
        using int64_batch = typename T::int64_batch;
        using float_batch = typename T::float_batch;
        using double_batch = typename T::double_batch;
        using int32_vector = typename T::int32_vector;
        using int64_vector = typename T::int64_vector;
        using float_vector = typename T::float_vector;
        using double_vector = typename T::double_vector;
        using char_vector = typename T::char_vector;
        using uchar_vector = typename T::uchar_vector;

        int32_batch i32bres;
        int64_batch i64bres;
        float_batch fbres;
        double_batch dbres;

        constexpr std::size_t fsize = float_batch::size;
        constexpr std::size_t dsize = double_batch::size;
        int32_vector i32vres(fsize);
        int64_vector i64vres(fsize);
        float_vector fvres(fsize);
        double_vector dvres(fsize);
        char_vector cvres(fsize * 8);
        uchar_vector ucvres(fsize * 8);

        int32_vector i32vres2(dsize);
        int64_vector i64vres2(dsize);
        float_vector fvres2(dsize);
        double_vector dvres2(dsize);
        char_vector cvres2(dsize * 8, char(0));
        using uchar = unsigned char;
        uchar_vector ucvres2(dsize * 8, uchar(0));

        bool success = true;
        bool tmp_success = true;

        std::string name = tester.name;
        std::string name_shift = std::string(name.size(), '-');
        std::string dash(8, '-');
        std::string space(8, ' ');

        out << dash << name_shift << dash << std::endl;
        out << space << name << space << std::endl;
        out << dash << name_shift << dash << std::endl
            << std::endl;

        // float

        std::string topic = "store float   -> float  : ";
        fbres = load_simd<float>(tester.f_vec.data(), aligned_mode());
        store_simd(fvres.data(), fbres, aligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;
        
        topic = "storeu float  -> float  : ";
        fbres = load_simd<float>(tester.f_vec.data(), unaligned_mode());
        store_simd(fvres.data(), fbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "store float   -> int32  : ";
        fbres = load_simd<float>(tester.f_vec.data(), aligned_mode());
        store_simd<int32_t, float>(i32vres.data(), fbres, aligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "storeu float  -> int32  : ";
        fbres = load_simd<float>(tester.f_vec.data(), unaligned_mode());
        store_simd<int32_t, float>(i32vres.data(), fbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "store float   -> int64  : ";
        fbres = load_simd<float>(tester.f_vec.data(), aligned_mode());
        store_simd<int64_t, float>(i64vres.data(), fbres, aligned_mode());
        tmp_success = check_almost_equal(topic, i64vres, tester.i64_vec, out);
        success = tmp_success && success;

        topic = "storeu float  -> int64  : ";
        fbres = load_simd<float>(tester.f_vec.data(), unaligned_mode());
        store_simd<int64_t, float>(i64vres.data(), fbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i64vres, tester.i64_vec, out);
        success = tmp_success && success;

        topic = "store float   -> double : ";
        fbres = load_simd<float>(tester.f_vec.data(), aligned_mode());
        store_simd<double, float>(dvres.data(), fbres, aligned_mode());
        tmp_success = check_almost_equal(topic, dvres, tester.d_vec, out);
        success = tmp_success && success;

        topic = "storeu float  -> double : ";
        fbres = load_simd<float>(tester.f_vec.data(), unaligned_mode());
        store_simd<double, float>(dvres.data(), fbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, dvres, tester.d_vec, out);
        success = tmp_success && success;

        topic = "store float   -> char   : ";
        fbres = load_simd<float>(tester.f_vec.data(), aligned_mode());
        store_simd<char, float>(cvres.data(), fbres, aligned_mode());
        std::copy(tester.c_vec.cbegin() + fsize, tester.c_vec.cend(), cvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, cvres, tester.c_vec, out);
        success = tmp_success && success;

        topic = "storeu float  -> char   : ";
        fbres = load_simd<float>(tester.f_vec.data(), unaligned_mode());
        store_simd<char, float>(cvres.data(), fbres, unaligned_mode());
        std::copy(tester.c_vec.cbegin() + fsize, tester.c_vec.cend(), cvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, cvres, tester.c_vec, out);
        success = tmp_success && success;

        topic = "store float   -> uchar  : ";
        fbres = load_simd<float>(tester.f_vec.data(), aligned_mode());
        store_simd<unsigned char, float>(ucvres.data(), fbres, aligned_mode());
        std::copy(tester.uc_vec.cbegin() + fsize, tester.uc_vec.cend(), ucvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, ucvres, tester.uc_vec, out);
        success = tmp_success && success;

        topic = "storeu float  -> uchar  : ";
        fbres = load_simd<float>(tester.f_vec.data(), unaligned_mode());
        store_simd<unsigned char, float>(ucvres.data(), fbres, unaligned_mode());
        std::copy(tester.uc_vec.cbegin() + fsize, tester.uc_vec.cend(), ucvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, ucvres, tester.uc_vec, out);
        success = tmp_success && success;

        // double

        topic = "store double  -> float  : ";
        dbres = load_simd<double>(tester.d_vec.data(), aligned_mode());
        store_simd<float, double>(fvres2.data(), dbres, aligned_mode());
        tmp_success = check_almost_equal(topic, fvres2, tester.f_vec2, out);
        success = tmp_success && success;

        topic = "storeu double -> float  : ";
        dbres = load_simd<double>(tester.d_vec.data(), unaligned_mode());
        store_simd<float, double>(fvres2.data(), dbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, fvres2, tester.f_vec2, out);
        success = tmp_success && success;

        topic = "store double  -> int32  : ";
        dbres = load_simd<double>(tester.d_vec.data(), aligned_mode());
        store_simd<int32_t, double>(i32vres2.data(), dbres, aligned_mode());
        tmp_success = check_almost_equal(topic, i32vres2, tester.i32_vec2, out);
        success = tmp_success && success;

        topic = "storeu double -> int32  : ";
        dbres = load_simd<double>(tester.d_vec.data(), unaligned_mode());
        store_simd<int32_t, double>(i32vres2.data(), dbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i32vres2, tester.i32_vec2, out);
        success = tmp_success && success;

        topic = "store double  -> int64  : ";
        dbres = load_simd<double>(tester.d_vec.data(), aligned_mode());
        store_simd<int64_t, double>(i64vres2.data(), dbres, aligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "storeu double -> int64  : ";
        dbres = load_simd<double>(tester.d_vec.data(), unaligned_mode());
        store_simd<int64_t, double>(i64vres2.data(), dbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "store double  -> double : ";
        dbres = load_simd<double>(tester.d_vec.data(), aligned_mode());
        store_simd(dvres2.data(), dbres, aligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "storeu double -> double : ";
        dbres = load_simd<double>(tester.d_vec.data(), unaligned_mode());
        store_simd(dvres2.data(), dbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "store double  -> char   : ";
        dbres = load_simd<double>(tester.d_vec.data(), aligned_mode());
        store_simd<char, double>(cvres2.data(), dbres, aligned_mode());
        std::copy(tester.c_vec2.cbegin() + dsize, tester.c_vec2.cend(), cvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, cvres2, tester.c_vec2, out);
        success = tmp_success && success;

        topic = "storeu double -> char   : ";
        dbres = load_simd<double>(tester.d_vec.data(), unaligned_mode());
        store_simd<char, double>(cvres2.data(), dbres, unaligned_mode());
        std::copy(tester.c_vec2.cbegin() + dsize, tester.c_vec2.cend(), cvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, cvres2, tester.c_vec2, out);
        success = tmp_success && success;

        topic = "store double  -> uchar  : ";
        dbres = load_simd<double>(tester.d_vec.data(), aligned_mode());
        store_simd<unsigned char, double>(ucvres2.data(), dbres, aligned_mode());
        std::copy(tester.uc_vec2.cbegin() + dsize, tester.uc_vec2.cend(), ucvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, ucvres2, tester.uc_vec2, out);
        success = tmp_success && success;

        topic = "storeu double -> uchar  : ";
        dbres = load_simd<double>(tester.d_vec.data(), unaligned_mode());
        store_simd<unsigned char, double>(ucvres2.data(), dbres, unaligned_mode());
        std::copy(tester.uc_vec2.cbegin() + dsize, tester.uc_vec2.cend(), ucvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, ucvres2, tester.uc_vec2, out);
        success = tmp_success && success;

        // int32

        topic = "store int32   -> float  : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), aligned_mode());
        store_simd<float, int32_t>(fvres.data(), i32bres, aligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "storeu int32  -> float  : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), unaligned_mode());
        store_simd<float, int32_t>(fvres.data(), i32bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "store int32   -> int32  : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), aligned_mode());
        store_simd(i32vres.data(), i32bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "storeu int32  -> int32  : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), unaligned_mode());
        store_simd(i32vres.data(), i32bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "store int32   -> int64  : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), aligned_mode());
        store_simd<int64_t, int32_t>(i64vres.data(), i32bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i64vres, tester.i64_vec, out);
        success = tmp_success && success;

        topic = "storeu int32  -> int64  : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), unaligned_mode());
        store_simd<int64_t, int32_t>(i64vres.data(), i32bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i64vres, tester.i64_vec, out);
        success = tmp_success && success;

        topic = "store int32   -> double : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), aligned_mode());
        store_simd<double, int32_t>(dvres.data(), i32bres, aligned_mode());
        tmp_success = check_almost_equal(topic, dvres, tester.d_vec, out);
        success = tmp_success && success;

        topic = "storeu int32  -> double : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), unaligned_mode());
        store_simd<double, int32_t>(dvres.data(), i32bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, dvres, tester.d_vec, out);
        success = tmp_success && success;

        topic = "store int32   -> char   : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), aligned_mode());
        store_simd<char, int32_t>(cvres.data(), i32bres, aligned_mode());
        std::copy(tester.c_vec.cbegin() + fsize, tester.c_vec.cend(), cvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, cvres, tester.c_vec, out);
        success = tmp_success && success;

        topic = "storeu int32  -> char   : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), unaligned_mode());
        store_simd<char, int32_t>(cvres.data(), i32bres, unaligned_mode());
        std::copy(tester.c_vec.cbegin() + fsize, tester.c_vec.cend(), cvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, cvres, tester.c_vec, out);
        success = tmp_success && success;

        topic = "store int32   -> uchar  : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), aligned_mode());
        store_simd<unsigned char, int32_t>(ucvres.data(), i32bres, aligned_mode());
        std::copy(tester.uc_vec.cbegin() + fsize, tester.uc_vec.cend(), ucvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, ucvres, tester.uc_vec, out);
        success = tmp_success && success;

        topic = "storeu int32  -> uchar  : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), unaligned_mode());
        store_simd<unsigned char, int32_t>(ucvres.data(), i32bres, unaligned_mode());
        std::copy(tester.uc_vec.cbegin() + fsize, tester.uc_vec.cend(), ucvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, ucvres, tester.uc_vec, out);
        success = tmp_success && success;

        // int64

        topic = "store int64   -> float  : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), aligned_mode());
        store_simd<float, int64_t>(fvres2.data(), i64bres, aligned_mode());
        tmp_success = check_almost_equal(topic, fvres2, tester.f_vec2, out);
        success = tmp_success && success;

        topic = "storeu int64  -> float  : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), unaligned_mode());
        store_simd<float, int64_t>(fvres2.data(), i64bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, fvres2, tester.f_vec2, out);
        success = tmp_success && success;

        topic = "store int64   -> int32  : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), aligned_mode());
        store_simd<int32_t, int64_t>(i32vres2.data(), i64bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i32vres2, tester.i32_vec2, out);
        success = tmp_success && success;

        topic = "storeu int64  -> int32  : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), unaligned_mode());
        store_simd<int32_t, int64_t>(i32vres2.data(), i64bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i32vres2, tester.i32_vec2, out);
        success = tmp_success && success;

        topic = "store int64   -> int64  : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), aligned_mode());
        store_simd(i64vres2.data(), i64bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "storeu int64  -> int64  : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), unaligned_mode());
        store_simd(i64vres2.data(), i64bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "store int64   -> double : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), aligned_mode());
        store_simd<double, int64_t>(dvres2.data(), i64bres, aligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "storeu int64  -> double : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), unaligned_mode());
        store_simd<double, int64_t>(dvres2.data(), i64bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "store int64   -> char   : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), aligned_mode());
        store_simd<char, int64_t>(cvres2.data(), i64bres, aligned_mode());
        std::copy(tester.c_vec2.cbegin() + dsize, tester.c_vec2.cend(), cvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, cvres2, tester.c_vec2, out);
        success = tmp_success && success;

        topic = "storeu int64  -> char   : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), unaligned_mode());
        store_simd<char, int64_t>(cvres2.data(), i64bres, unaligned_mode());
        std::copy(tester.c_vec2.cbegin() + dsize, tester.c_vec2.cend(), cvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, cvres2, tester.c_vec2, out);
        success = tmp_success && success;

        topic = "store int64   -> uchar  : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), aligned_mode());
        store_simd<unsigned char, int64_t>(ucvres2.data(), i64bres, aligned_mode());
        std::copy(tester.uc_vec2.cbegin() + dsize, tester.uc_vec2.cend(), ucvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, ucvres2, tester.uc_vec2, out);
        success = tmp_success && success;

        topic = "storeu int64  -> uchar  : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), unaligned_mode());
        store_simd<unsigned char, int64_t>(ucvres2.data(), i64bres, unaligned_mode());
        std::copy(tester.uc_vec2.cbegin() + dsize, tester.uc_vec2.cend(), ucvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, ucvres2, tester.uc_vec2, out);
        success = tmp_success && success;

        return success;
    }

    /*****************************
     * complex load / store test *
     *****************************/

    template <class T>
    inline bool test_simd_complex_api(std::ostream& out, T& tester)
    {
        using batch_type = typename T::batch_type;
        using value_type = typename batch_type::value_type;
        using real_batch_type = typename T::real_batch_type;
        using float_vector = typename T::float_vector;
        using double_vector = typename T::double_vector;
        using float_complex_vector = typename T::float_complex_vector;
        using double_complex_vector = typename T::double_complex_vector;

        bool success = true;
        bool tmp_success = true;

        std::string name = tester.name;
        std::string name_shift = std::string(name.size(), '-');
        std::string dash(8, '-');
        std::string space(8, ' ');

        out << dash << name_shift << dash << std::endl;
        out << space << name << space << std::endl;
        out << dash << name_shift << dash << std::endl
            << std::endl;

        std::string topic = "load float complex    : ";
        batch_type fref, fres;
        fref = load_simd<float, value_type>(tester.f_vec_real.data(), tester.f_vec_imag.data(), aligned_mode());
        fres.load_aligned(tester.fc_vec.data());
        tmp_success = all(fres.real() == fref.real()) && all(fres.imag() == fref.imag());
        success = tmp_success && success;

        topic = "loadu float complex   : ";
        fref = load_simd<float, value_type>(tester.f_vec_real.data(), tester.f_vec_imag.data(), unaligned_mode());
        fres.load_aligned(tester.fc_vec.data());
        tmp_success = all(fres.real() == fref.real()) && all(fres.imag() == fref.imag());
        success = tmp_success && success;

        topic = "load double complex   : ";
        batch_type dref, dres;
        dref = load_simd<double, value_type>(tester.d_vec_real.data(), tester.d_vec_imag.data(), aligned_mode());
        dres.load_aligned(tester.dc_vec.data());
        tmp_success = all(dres.real() == dref.real()) && all(dres.imag() == dref.imag());
        success = tmp_success && success;

        topic = "loadu double complex  : ";
        dref = load_simd<double, value_type>(tester.d_vec_real.data(), tester.d_vec_imag.data(), unaligned_mode());
        dres.load_aligned(tester.dc_vec.data());
        tmp_success = all(dres.real() == dref.real()) && all(dres.imag() == dref.imag());
        success = tmp_success && success;

        topic = "store float complex   : ";
        float_vector fc_res_real(tester.f_vec_real.size());
        float_vector fc_res_imag(tester.f_vec_imag.size());
        store_simd<float, value_type>(fc_res_real.data(), fc_res_imag.data(), fres, aligned_mode());
        tmp_success = check_almost_equal(topic, fc_res_real, tester.f_vec_real, out);
        tmp_success = check_almost_equal(topic, fc_res_imag, tester.f_vec_imag, out);
        success = tmp_success && success;

        topic = "storeu float complex  : ";
        store_simd<float, value_type>(fc_res_real.data(), fc_res_imag.data(), fres, unaligned_mode());
        tmp_success = check_almost_equal(topic, fc_res_real, tester.f_vec_real, out);
        tmp_success = check_almost_equal(topic, fc_res_imag, tester.f_vec_imag, out);
        success = tmp_success && success;

        topic = "store double complex  : ";
        double_vector dc_res_real(tester.d_vec_real.size());
        double_vector dc_res_imag(tester.d_vec_imag.size());
        store_simd<double, value_type>(dc_res_real.data(), dc_res_imag.data(), dres, aligned_mode());
        tmp_success = check_almost_equal(topic, dc_res_real, tester.d_vec_real, out);
        tmp_success = check_almost_equal(topic, dc_res_imag, tester.d_vec_imag, out);
        success = tmp_success && success;

        topic = "storeu double complex : ";
        store_simd<double, value_type>(dc_res_real.data(), dc_res_imag.data(), dres, aligned_mode());
        tmp_success = check_almost_equal(topic, dc_res_real, tester.d_vec_real, out);
        tmp_success = check_almost_equal(topic, dc_res_imag, tester.d_vec_imag, out);
        success = tmp_success && success;

        topic = "load float complex r  : ";
        fref = load_simd<float, value_type>(tester.f_vec_real.data(), tester.f_vec_zero.data(), aligned_mode());
        fres = load_simd<float, value_type>(tester.f_vec_real.data(), aligned_mode());
        tmp_success = all(fres.real() == fref.real()) && all(fres.imag() == fref.imag());
        success = tmp_success && success;

        topic = "loadu float complex r : ";
        fref = load_simd<float, value_type>(tester.f_vec_real.data(), tester.f_vec_zero.data(), unaligned_mode());
        fres = load_simd<float, value_type>(tester.f_vec_real.data(), unaligned_mode());
        tmp_success = all(fres.real() == fref.real()) && all(fres.imag() == fref.imag());
        success = tmp_success && success;

        topic = "load double complex r : ";
        dref = load_simd<double, value_type>(tester.d_vec_real.data(), tester.d_vec_zero.data(), aligned_mode());
        dres = load_simd<double, value_type>(tester.d_vec_real.data(), aligned_mode());
        tmp_success = all(dres.real() == dref.real()) && all(dres.imag() == dref.imag());
        success = tmp_success && success;

        topic = "loadu double complex r: ";
        dref = load_simd<double, value_type>(tester.d_vec_real.data(), tester.d_vec_zero.data(), unaligned_mode());
        dres = load_simd<double, value_type>(tester.d_vec_real.data(), unaligned_mode());
        tmp_success = all(dres.real() == dref.real()) && all(dres.imag() == dref.imag());
        success = tmp_success && success;

        return success;
    }
}

#endif

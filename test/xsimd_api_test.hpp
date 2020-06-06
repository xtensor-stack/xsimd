/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
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
    /*********************
     * load_store tester *
     *********************/

    template <std::size_t N, std::size_t A>
    struct simd_api_load_store_tester
    {
        using int32_batch = batch<int32_t, N * 2>;
        using int64_batch = batch<int64_t, N>;
        using float_batch = batch<float, N * 2>;
        using double_batch = batch<double, N>;

        using char_vector = std::vector<char, aligned_allocator<char, A>>;
        using int8_vector = std::vector<int8_t, aligned_allocator<int8_t, A>>;
        using uint8_vector = std::vector<uint8_t, aligned_allocator<uint8_t, A>>;
        using int16_vector = std::vector<int16_t, aligned_allocator<int16_t, A>>;
        using uint16_vector = std::vector<uint16_t, aligned_allocator<uint16_t, A>>;
        using int32_vector = std::vector<int32_t, aligned_allocator<int32_t, A>>;
        using uint32_vector = std::vector<uint32_t, aligned_allocator<uint32_t, A>>;
        using int64_vector = std::vector<int64_t, aligned_allocator<int64_t, A>>;
        using uint64_vector = std::vector<uint64_t, aligned_allocator<uint64_t, A>>;
#ifdef XSIMD_32_BIT_ABI
        using long_vector = std::vector<long, aligned_allocator<long, A>>;
        using ulong_vector = std::vector<unsigned long, aligned_allocator<unsigned long, A>>;
#endif
        using float_vector = std::vector<float, aligned_allocator<float, A>>;
        using double_vector = std::vector<double, aligned_allocator<double, A>>;

        std::string name;

        char_vector char_vec;
        int8_vector i8_vec;
        uint8_vector ui8_vec;
        int16_vector i16_vec;
        uint16_vector ui16_vec;
        int32_vector i32_vec;
        uint32_vector ui32_vec;
        int64_vector i64_vec;
        uint64_vector ui64_vec;
        float_vector f_vec;
        double_vector d_vec;

        char_vector char_vec2;
        int8_vector i8_vec2;
        uint8_vector ui8_vec2;
        int16_vector i16_vec2;
        uint16_vector ui16_vec2;
        int32_vector i32_vec2;
        uint32_vector ui32_vec2;
        int64_vector i64_vec2;
        uint64_vector ui64_vec2;
        float_vector f_vec2;
        double_vector d_vec2;

#ifdef XSIMD_32_BIT_ABI
        long_vector long_vec;
        ulong_vector ulong_vec;
        long_vector long_vec2;
        ulong_vector ulong_vec2;
#endif
        bool bool_vec[16 * N];
        bool bool_vec2[8 * N];

        simd_api_load_store_tester(const std::string& n);
    };

    template <std::size_t N, std::size_t A>
    inline simd_api_load_store_tester<N, A>::simd_api_load_store_tester(const std::string& n)
        : name(n),
          char_vec(16 * N), i8_vec(16 * N), ui8_vec(16 * N), i16_vec(16 * N), ui16_vec(16 * N),
          i32_vec(2 * N), ui32_vec(2 * N), i64_vec(2 * N), ui64_vec(2 * N), f_vec(2 * N), d_vec(2 * N),
          char_vec2(8 * N), i8_vec2(8 * N), ui8_vec2(8 * N), i16_vec2(8 * N), ui16_vec2(8 * N),
          i32_vec2(N), ui32_vec2(N), i64_vec2(N), ui64_vec2(N), f_vec2(N), d_vec2(N)
    {
        std::iota(char_vec.begin(), char_vec.end(), char(1));
        std::iota(i8_vec.begin(), i8_vec.end(), int8_t(1));
        std::iota(ui8_vec.begin(), ui8_vec.end(), uint8_t(1));
        std::iota(i16_vec.begin(), i16_vec.end(), int16_t(1));
        std::iota(ui16_vec.begin(), ui16_vec.end(), uint16_t(1));
        std::iota(i32_vec.begin(), i32_vec.end(), int32_t(1));
        std::iota(ui32_vec.begin(), ui32_vec.end(), uint32_t(1));
        std::iota(i64_vec.begin(), i64_vec.end(), int64_t(1));
        std::iota(ui64_vec.begin(), ui64_vec.end(), uint64_t(1));
        std::iota(f_vec.begin(), f_vec.end(), float(1));
        std::iota(d_vec.begin(), d_vec.end(), double(1));
        std::iota(char_vec2.begin(), char_vec2.end(), char(1));
        std::iota(i8_vec2.begin(), i8_vec2.end(), int8_t(1));
        std::iota(ui8_vec2.begin(), ui8_vec2.end(), uint8_t(1));
        std::iota(i16_vec2.begin(), i16_vec2.end(), int16_t(1));
        std::iota(ui16_vec2.begin(), ui16_vec2.end(), uint16_t(1));
        std::iota(i32_vec2.begin(), i32_vec2.end(), int32_t(1));
        std::iota(ui32_vec2.begin(), ui32_vec2.end(), uint32_t(1));
        std::iota(i64_vec2.begin(), i64_vec2.end(), int64_t(1));
        std::iota(ui64_vec2.begin(), ui64_vec2.end(), uint64_t(1));
        std::iota(f_vec2.begin(), f_vec2.end(), float(1));
        std::iota(d_vec2.begin(), d_vec2.end(), double(1));

#ifdef XSIMD_32_BIT_ABI
        using ulong = unsigned long;
        long_vec.resize(2 * N);
        ulong_vec.resize(2 * N);
        long_vec2.resize(N);
        ulong_vec2.resize(N);
        std::iota(long_vec.begin(), long_vec.end(), long(1));
        std::iota(ulong_vec.begin(), ulong_vec.end(), ulong(1));
        std::iota(long_vec2.begin(), long_vec2.end(), long(1));
        std::iota(ulong_vec2.begin(), ulong_vec2.end(), ulong(1));
#endif
        for (size_t i = 0; i < 8 * N; i += 2)
        {
            bool_vec[i] = true;
            bool_vec[i+1] = false;
            bool_vec2[i] = true;
            bool_vec2[i+1] = false;
            bool_vec[8*N + i] = true;
            bool_vec[8*N + i+1] = false;
        }
    }

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
        using char_vector = typename T::char_vector;
        using int8_vector = typename T::int8_vector;
        using uint8_vector = typename T::uint8_vector;
        using int32_vector = typename T::int32_vector;
        using int64_vector = typename T::int64_vector;
        using float_vector = typename T::float_vector;
        using double_vector = typename T::double_vector;

        int32_batch i32bres;
        int64_batch i64bres;
        float_batch fbres;
        double_batch dbres;

        char_vector ccvres(float_batch::size);
        int8_vector cvres(float_batch::size);
        uint8_vector ucvres(float_batch::size);
        int32_vector i32vres(float_batch::size);
        int64_vector i64vres(float_batch::size);
        float_vector fvres(float_batch::size);
        double_vector dvres(float_batch::size);

        char_vector ccvres2(float_batch::size);
        int8_vector cvres2(float_batch::size);
        uint8_vector ucvres2(float_batch::size);
        int32_vector i32vres2(double_batch::size);
        int64_vector i64vres2(double_batch::size);
        float_vector fvres2(double_batch::size);
        double_vector dvres2(double_batch::size);

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
        fbres = load_simd<char, float>(tester.char_vec.data(), aligned_mode());
        store_simd(fvres.data(), fbres, aligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "loadu char   -> float  : ";
        fbres = load_simd<char, float>(tester.char_vec.data(), unaligned_mode());
        store_simd(fvres.data(), fbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "load int8    -> float  : ";
        fbres = load_simd<int8_t, float>(tester.i8_vec.data(), aligned_mode());
        store_simd(fvres.data(), fbres, aligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "loadu int8   -> float  : ";
        fbres = load_simd<int8_t, float>(tester.i8_vec.data(), unaligned_mode());
        store_simd(fvres.data(), fbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "load uint8   -> float  : ";
        fbres = load_simd<uint8_t, float>(tester.ui8_vec.data(), aligned_mode());
        store_simd(fvres.data(), fbres, aligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "loadu uint8  -> float  : ";
        fbres = load_simd<uint8_t, float>(tester.ui8_vec.data(), unaligned_mode());
        store_simd(fvres.data(), fbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, fvres, tester.f_vec, out);
        success = tmp_success && success;

        topic = "load bool    -> float  : ";
        fbres = load_simd<bool, float>(tester.bool_vec, aligned_mode());
        store_simd(fvres.data(), fbres, aligned_mode());

        topic = "loadu bool   -> float  : ";
        fbres = load_simd<bool, float>(tester.bool_vec, aligned_mode());
        store_simd(fvres.data(), fbres, aligned_mode());
        
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
        dbres = load_simd<char, double>(tester.char_vec.data(), aligned_mode());
        store_simd(dvres2.data(), dbres, aligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "loadu char   -> double : ";
        dbres = load_simd<char, double>(tester.char_vec.data(), unaligned_mode());
        store_simd(dvres2.data(), dbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "load int8    -> double : ";
        dbres = load_simd<int8_t, double>(tester.i8_vec.data(), aligned_mode());
        store_simd(dvres2.data(), dbres, aligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "loadu int8   -> double : ";
        dbres = load_simd<int8_t, double>(tester.i8_vec.data(), unaligned_mode());
        store_simd(dvres2.data(), dbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "load uint8   -> double : ";
        dbres = load_simd<uint8_t, double>(tester.ui8_vec.data(), aligned_mode());
        store_simd(dvres2.data(), dbres, aligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "loadu uint8  -> double : ";
        dbres = load_simd<uint8_t, double>(tester.ui8_vec.data(), unaligned_mode());
        store_simd(dvres2.data(), dbres, unaligned_mode());
        tmp_success = check_almost_equal(topic, dvres2, tester.d_vec2, out);
        success = tmp_success && success;

        topic = "load bool    -> double : ";
        dbres = load_simd<bool, double>(tester.bool_vec, aligned_mode());
        store_simd(dvres2.data(), dbres, aligned_mode());

        topic = "loadu bool   -> double : ";
        dbres = load_simd<bool, double>(tester.bool_vec, aligned_mode());
        store_simd(dvres2.data(), dbres, aligned_mode());
        
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
        i32bres = load_simd<char, int32_t>(tester.char_vec.data(), aligned_mode());
        store_simd(i32vres.data(), i32bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "loadu char8   -> int32  : ";
        i32bres = load_simd<char, int32_t>(tester.char_vec.data(), unaligned_mode());
        store_simd(i32vres.data(), i32bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "load int8    -> int32  : ";
        i32bres = load_simd<int8_t, int32_t>(tester.i8_vec.data(), aligned_mode());
        store_simd(i32vres.data(), i32bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "loadu int8   -> int32  : ";
        i32bres = load_simd<int8_t, int32_t>(tester.i8_vec.data(), unaligned_mode());
        store_simd(i32vres.data(), i32bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "load uint8   -> int32  : ";
        i32bres = load_simd<uint8_t, int32_t>(tester.ui8_vec.data(), aligned_mode());
        store_simd(i32vres.data(), i32bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "loadu uint8  -> int32  : ";
        i32bres = load_simd<uint8_t, int32_t>(tester.ui8_vec.data(), unaligned_mode());
        store_simd(i32vres.data(), i32bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i32vres, tester.i32_vec, out);
        success = tmp_success && success;

        topic = "load bool    -> int32  : ";
        i32bres = load_simd<bool, int32_t>(tester.bool_vec, aligned_mode());
        store_simd(i32vres.data(), i32bres, aligned_mode());

        topic = "loadu bool   -> int32  : ";
        i32bres = load_simd<bool, int32_t>(tester.bool_vec, unaligned_mode());
        store_simd(i32vres.data(), i32bres, unaligned_mode());

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
        i64bres = load_simd<char, int64_t>(tester.char_vec.data(), aligned_mode());
        store_simd(i64vres2.data(), i64bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "loadu char   -> int64  : ";
        i64bres = load_simd<char, int64_t>(tester.char_vec.data(), unaligned_mode());
        store_simd(i64vres2.data(), i64bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "load int8    -> int64  : ";
        i64bres = load_simd<int8_t, int64_t>(tester.i8_vec.data(), aligned_mode());
        store_simd(i64vres2.data(), i64bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "loadu int8   -> int64  : ";
        i64bres = load_simd<int8_t, int64_t>(tester.i8_vec.data(), unaligned_mode());
        store_simd(i64vres2.data(), i64bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "load uint8   -> int64  : ";
        i64bres = load_simd<uint8_t, int64_t>(tester.ui8_vec.data(), aligned_mode());
        store_simd(i64vres2.data(), i64bres, aligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "loadu uint8  -> int64  : ";
        i64bres = load_simd<uint8_t, int64_t>(tester.ui8_vec.data(), unaligned_mode());
        store_simd(i64vres2.data(), i64bres, unaligned_mode());
        tmp_success = check_almost_equal(topic, i64vres2, tester.i64_vec2, out);
        success = tmp_success && success;

        topic = "load bool    -> int64  : ";
        i64bres = load_simd<bool, int64_t>(tester.bool_vec, aligned_mode());
        store_simd(i64vres2.data(), i64bres, aligned_mode());

        topic = "loadu bool   -> int64  : ";
        i64bres = load_simd<bool, int64_t>(tester.bool_vec, unaligned_mode());
        store_simd(i64vres2.data(), i64bres, unaligned_mode());

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
        using int8_vector = typename T::int8_vector;
        using uint8_vector = typename T::uint8_vector;

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
        char_vector ccvres(fsize * 8);
        int8_vector cvres(fsize * 8);
        uint8_vector ucvres(fsize * 8);
        bool bvres[fsize * 8];

        int32_vector i32vres2(dsize);
        int64_vector i64vres2(dsize);
        float_vector fvres2(dsize);
        double_vector dvres2(dsize);
        char_vector ccvres2(dsize * 8, char(0));
        int8_vector cvres2(dsize * 8, int8_t(0));
        uint8_vector ucvres2(dsize * 8, uint8_t(0));
        bool bvres2[dsize * 8];

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
        store_simd<char, float>(ccvres.data(), fbres, aligned_mode());
        std::copy(tester.char_vec.cbegin() + fsize, tester.char_vec.cend(), ccvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, ccvres, tester.char_vec, out);
        success = tmp_success && success;

        topic = "storeu float  -> char   : ";
        fbres = load_simd<float>(tester.f_vec.data(), unaligned_mode());
        store_simd<char, float>(ccvres.data(), fbres, unaligned_mode());
        std::copy(tester.char_vec.cbegin() + fsize, tester.char_vec.cend(), ccvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, ccvres, tester.char_vec, out);
        success = tmp_success && success;

        topic = "store float   -> int8   : ";
        fbres = load_simd<float>(tester.f_vec.data(), aligned_mode());
        store_simd<int8_t, float>(cvres.data(), fbres, aligned_mode());
        std::copy(tester.i8_vec.cbegin() + fsize, tester.i8_vec.cend(), cvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, cvres, tester.i8_vec, out);
        success = tmp_success && success;

        topic = "storeu float  -> int8   : ";
        fbres = load_simd<float>(tester.f_vec.data(), unaligned_mode());
        store_simd<int8_t, float>(cvres.data(), fbres, unaligned_mode());
        std::copy(tester.i8_vec.cbegin() + fsize, tester.i8_vec.cend(), cvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, cvres, tester.i8_vec, out);
        success = tmp_success && success;

        topic = "store float   -> uint8  : ";
        fbres = load_simd<float>(tester.f_vec.data(), aligned_mode());
        store_simd<uint8_t, float>(ucvres.data(), fbres, aligned_mode());
        std::copy(tester.ui8_vec.cbegin() + fsize, tester.ui8_vec.cend(), ucvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, ucvres, tester.ui8_vec, out);
        success = tmp_success && success;

        topic = "storeu float  -> uint8  : ";
        fbres = load_simd<float>(tester.f_vec.data(), unaligned_mode());
        store_simd<uint8_t, float>(ucvres.data(), fbres, unaligned_mode());
        std::copy(tester.ui8_vec.cbegin() + fsize, tester.ui8_vec.cend(), ucvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, ucvres, tester.ui8_vec, out);
        success = tmp_success && success;

        topic = "store float   -> bool   : ";
        fbres = load_simd<float>(tester.f_vec.data(), aligned_mode());
        store_simd<bool, float>(bvres, fbres, aligned_mode());

        topic = "storeu float  -> bool   : ";
        fbres = load_simd<float>(tester.f_vec.data(), unaligned_mode());
        store_simd<bool, float>(bvres, fbres, unaligned_mode());

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
        store_simd<char, double>(ccvres2.data(), dbres, aligned_mode());
        std::copy(tester.char_vec2.cbegin() + dsize, tester.char_vec2.cend(), ccvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, ccvres2, tester.char_vec2, out);
        success = tmp_success && success;

        topic = "storeu double -> char   : ";
        dbres = load_simd<double>(tester.d_vec.data(), unaligned_mode());
        store_simd<char, double>(ccvres2.data(), dbres, unaligned_mode());
        std::copy(tester.char_vec2.cbegin() + dsize, tester.char_vec2.cend(), ccvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, ccvres2, tester.char_vec2, out);
        success = tmp_success && success;

        topic = "store double  -> int8   : ";
        dbres = load_simd<double>(tester.d_vec.data(), aligned_mode());
        store_simd<int8_t, double>(cvres2.data(), dbres, aligned_mode());
        std::copy(tester.i8_vec2.cbegin() + dsize, tester.i8_vec2.cend(), cvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, cvres2, tester.i8_vec2, out);
        success = tmp_success && success;

        topic = "storeu double -> int8   : ";
        dbres = load_simd<double>(tester.d_vec.data(), unaligned_mode());
        store_simd<int8_t, double>(cvres2.data(), dbres, unaligned_mode());
        std::copy(tester.i8_vec2.cbegin() + dsize, tester.i8_vec2.cend(), cvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, cvres2, tester.i8_vec2, out);
        success = tmp_success && success;

        topic = "store double  -> uint8  : ";
        dbres = load_simd<double>(tester.d_vec.data(), aligned_mode());
        store_simd<uint8_t, double>(ucvres2.data(), dbres, aligned_mode());
        std::copy(tester.ui8_vec2.cbegin() + dsize, tester.ui8_vec2.cend(), ucvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, ucvres2, tester.ui8_vec2, out);
        success = tmp_success && success;

        topic = "storeu double -> uint8  : ";
        dbres = load_simd<double>(tester.d_vec.data(), unaligned_mode());
        store_simd<uint8_t, double>(ucvres2.data(), dbres, unaligned_mode());
        std::copy(tester.ui8_vec2.cbegin() + dsize, tester.ui8_vec2.cend(), ucvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, ucvres2, tester.ui8_vec2, out);
        success = tmp_success && success;

        topic = "store double  -> bool   : ";
        dbres = load_simd<double>(tester.d_vec.data(), aligned_mode());
        store_simd<bool, double>(bvres2, dbres, aligned_mode());

        topic = "storeu double -> bool   : ";
        dbres = load_simd<double>(tester.d_vec.data(), unaligned_mode());
        store_simd<bool, double>(bvres2, dbres, unaligned_mode());

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
        store_simd<char, int32_t>(ccvres.data(), i32bres, aligned_mode());
        std::copy(tester.char_vec.cbegin() + fsize, tester.char_vec.cend(), ccvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, ccvres, tester.char_vec, out);
        success = tmp_success && success;

        topic = "storeu int32  -> char   : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), unaligned_mode());
        store_simd<char, int32_t>(ccvres.data(), i32bres, unaligned_mode());
        std::copy(tester.char_vec.cbegin() + fsize, tester.char_vec.cend(), ccvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, ccvres, tester.char_vec, out);
        success = tmp_success && success;

        topic = "store int32   -> uint8  : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), aligned_mode());
        store_simd<uint8_t, int32_t>(ucvres.data(), i32bres, aligned_mode());
        std::copy(tester.ui8_vec.cbegin() + fsize, tester.ui8_vec.cend(), ucvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, ucvres, tester.ui8_vec, out);
        success = tmp_success && success;

        topic = "storeu int32  -> uint8  : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), unaligned_mode());
        store_simd<uint8_t, int32_t>(ucvres.data(), i32bres, unaligned_mode());
        std::copy(tester.ui8_vec.cbegin() + fsize, tester.ui8_vec.cend(), ucvres.begin() + fsize);
        tmp_success = check_almost_equal(topic, ucvres, tester.ui8_vec, out);
        success = tmp_success && success;

        topic = "store int32   -> bool   : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), aligned_mode());
        store_simd<bool, int32_t>(bvres, i32bres, aligned_mode());

        topic = "storeu int32  -> bool   : ";
        i32bres = load_simd<int32_t>(tester.i32_vec.data(), unaligned_mode());
        store_simd<bool, int32_t>(bvres, i32bres, unaligned_mode());

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
        store_simd<char, int64_t>(ccvres2.data(), i64bres, aligned_mode());
        std::copy(tester.char_vec2.cbegin() + dsize, tester.char_vec2.cend(), ccvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, ccvres2, tester.char_vec2, out);
        success = tmp_success && success;

        topic = "storeu int64  -> char   : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), unaligned_mode());
        store_simd<char, int64_t>(ccvres2.data(), i64bres, unaligned_mode());
        std::copy(tester.char_vec2.cbegin() + dsize, tester.char_vec2.cend(), ccvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, ccvres2, tester.char_vec2, out);
        success = tmp_success && success;

        topic = "store int64   -> int8   : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), aligned_mode());
        store_simd<int8_t, int64_t>(cvres2.data(), i64bres, aligned_mode());
        std::copy(tester.i8_vec2.cbegin() + dsize, tester.i8_vec2.cend(), cvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, cvres2, tester.i8_vec2, out);
        success = tmp_success && success;

        topic = "storeu int64  -> int8   : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), unaligned_mode());
        store_simd<int8_t, int64_t>(cvres2.data(), i64bres, unaligned_mode());
        std::copy(tester.i8_vec2.cbegin() + dsize, tester.i8_vec2.cend(), cvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, cvres2, tester.i8_vec2, out);
        success = tmp_success && success;

        topic = "store int64   -> uint8_t  : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), aligned_mode());
        store_simd<uint8_t, int64_t>(ucvres2.data(), i64bres, aligned_mode());
        std::copy(tester.ui8_vec2.cbegin() + dsize, tester.ui8_vec2.cend(), ucvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, ucvres2, tester.ui8_vec2, out);
        success = tmp_success && success;

        topic = "storeu int64  -> uint8_t  : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), unaligned_mode());
        store_simd<uint8_t, int64_t>(ucvres2.data(), i64bres, unaligned_mode());
        std::copy(tester.ui8_vec2.cbegin() + dsize, tester.ui8_vec2.cend(), ucvres2.begin() + dsize);
        tmp_success = check_almost_equal(topic, ucvres2, tester.ui8_vec2, out);
        success = tmp_success && success;

        topic = "store int64   -> bool     : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), aligned_mode());
        store_simd<bool, int64_t>(bvres2, i64bres, aligned_mode());

        topic = "storeu int64  -> bool     : ";
        i64bres = load_simd<int64_t>(tester.i64_vec.data(), unaligned_mode());
        store_simd<bool, int64_t>(bvres2, i64bres, unaligned_mode());

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

    /**************
     * set tester *
     **************/

    template <class T>
    struct simd_api_set_tester
    {
        using batch_type = typename simd_traits<T>::type;
        using batch_bool_type = typename simd_traits<T>::bool_type;
        static constexpr size_t N = simd_traits<T>::size;
        using value_type = T;

        std::string name;
        T res;
        bool bool_res;

        simd_api_set_tester(const std::string& n);

        bool check_res(const batch_type& v) const;
        bool check_bool_res(const batch_bool_type& v) const;
    };

    template <class T>
    inline simd_api_set_tester<T>::simd_api_set_tester(const std::string& n)
        : name(n)
        , res(value_type(1))
        , bool_res(true)
    {
    }

    template <class T>
    inline bool simd_api_set_tester<T>::check_res(const batch_type& v) const
    {
        bool ret = true;
        for(std::size_t i = 0; i < N; ++i)
        {
            ret = ret && (v[i] == res);
        }
        return ret;
    }

    template <class T>
    inline bool simd_api_set_tester<T>::check_bool_res(const batch_bool_type& v) const
    {
        bool ret = true;
        for(size_t i = 0; i < N; ++i)
        {
            ret = ret && (v[i] == bool_res);
        }
        return ret;
    }

    template <class T>
    inline bool test_simd_api_set_impl(std::ostream& out, const T& tester)
    {
        bool success = true;
        bool tmp_success = true;

        auto tmp = set_simd(tester.res);
        tmp_success = tester.check_res(tmp);
        out << tester.name << " - set     : " << tmp_success;
        success = success && tmp_success;

        tmp_success = tester.check_bool_res(set_simd<bool, typename T::value_type>(tester.bool_res));
        out << tester.name << " - set bool: " << tmp_success;
        success = success && tmp_success;

        return tmp_success;
    }

#ifdef XSIMD_BATCH_DOUBLE_SIZE
    inline bool test_simd_api_set(std::ostream& out, const std::string& instr_name)
    {
        bool success = true;
        bool tmp_success = true;

        tmp_success = test_simd_api_set_impl(out, simd_api_set_tester<uint8_t>(instr_name + " uint8_t"));
        success = success && tmp_success;
        tmp_success = test_simd_api_set_impl(out, simd_api_set_tester<int8_t>(instr_name + " int8_t"));
        success = success && tmp_success;
        tmp_success = test_simd_api_set_impl(out, simd_api_set_tester<uint16_t>(instr_name + " uint16_t"));
        success = success && tmp_success;
        tmp_success = test_simd_api_set_impl(out, simd_api_set_tester<int16_t>(instr_name + " int8_t"));
        success = success && tmp_success;
        tmp_success = test_simd_api_set_impl(out, simd_api_set_tester<uint32_t>(instr_name + " uint32_t"));
        success = success && tmp_success;
        tmp_success = test_simd_api_set_impl(out, simd_api_set_tester<int32_t>(instr_name + " int32_t"));
        success = success && tmp_success;
        tmp_success = test_simd_api_set_impl(out, simd_api_set_tester<uint64_t>(instr_name + " uint64_t"));
        success = success && tmp_success;
        tmp_success = test_simd_api_set_impl(out, simd_api_set_tester<int64_t>(instr_name + " int64_t"));
        success = success && tmp_success;
        tmp_success = test_simd_api_set_impl(out, simd_api_set_tester<float>(instr_name + " float"));
        success = success && tmp_success;
        tmp_success = test_simd_api_set_impl(out, simd_api_set_tester<double>(instr_name + " double"));
        success = success && tmp_success;

        return success;
    }
#endif
}

#endif

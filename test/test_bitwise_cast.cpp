/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "test_utils.hpp"

template <class CP>
class bitwise_cast_test : public testing::Test
{
    static constexpr size_t N = CP::size;
    static constexpr size_t A = CP::alignment;

    template <class T>
    using batch_size = std::integral_constant<size_t, N * (sizeof(uint64_t) / sizeof(T))>;
    
    template <class T>
    using test_batch = xsimd::batch<T, batch_size<T>::value>;

    template <class T>
    using test_vector = std::vector<T, xsimd::aligned_allocator<T, A>>;

    template <class T>
    test_vector<T> make_vector(const test_batch<T>& b)
    {
        test_vector<T> v(test_batch<T>::size);
        b.store_aligned(v.data());
        return v;
    }

    template<typename FromT, typename ToT>
    void test_from_to(const char* from_name, const char* to_name)
    {
        using from_batch = test_batch<FromT>;
        using to_batch = test_batch<ToT>;

        auto input = from_batch(FromT(123));

        auto vector_input = make_vector(input);
        test_vector<ToT> reference_vector_output(to_batch::size);
        std::memcpy(reference_vector_output.data(), vector_input.data(), to_batch::size * sizeof(ToT));
        
        auto batch_cast_output = xsimd::bitwise_cast<to_batch>(input);
        auto batch_cast_vector_output = make_vector(batch_cast_output);

        EXPECT_VECTOR_EQ(batch_cast_vector_output, reference_vector_output) << print_function_name(std::string{} + from_name + " => " + to_name);
    }

protected:
    template<typename ToT>
    void test_to(const char* to_name)
    {
        test_from_to<int8_t, ToT>("int8_t", to_name);
        test_from_to<uint8_t, ToT>("uint8_t", to_name);
        test_from_to<int16_t, ToT>("int16_t", to_name);
        test_from_to<uint16_t, ToT>("uint16_t", to_name);
        test_from_to<int32_t, ToT>("int32_t", to_name);
        test_from_to<uint32_t, ToT>("uint32_t", to_name);
        test_from_to<int64_t, ToT>("int64_t", to_name);
        test_from_to<uint64_t, ToT>("uint64_t", to_name);
        test_from_to<float, ToT>("float", to_name);
        test_from_to<double, ToT>("double", to_name);
    }
};

TYPED_TEST_SUITE(bitwise_cast_test, conversion_types, conversion_test_names);

TYPED_TEST(bitwise_cast_test, to_int8)
{
    this->template test_to<int8_t>("int8_t");
}

TYPED_TEST(bitwise_cast_test, to_uint8)
{
    this->template test_to<uint8_t>("uint8_t");
}

TYPED_TEST(bitwise_cast_test, to_int16)
{
    this->template test_to<int16_t>("int16_t");
}

TYPED_TEST(bitwise_cast_test, to_uint16)
{
    this->template test_to<uint16_t>("uint16_t");
}

TYPED_TEST(bitwise_cast_test, to_int32)
{
    this->template test_to<int32_t>("int32_t");
}

TYPED_TEST(bitwise_cast_test, to_uint32)
{
    this->template test_to<uint32_t>("uint32_t");
}

TYPED_TEST(bitwise_cast_test, to_int64)
{
    this->template test_to<int64_t>("int64_t");
}

TYPED_TEST(bitwise_cast_test, to_uint64)
{
    this->template test_to<uint64_t>("uint64_t");
}

TYPED_TEST(bitwise_cast_test, to_float)
{
    this->template test_to<float>("float");
}

TYPED_TEST(bitwise_cast_test, to_double)
{
    this->template test_to<double>("double");
}

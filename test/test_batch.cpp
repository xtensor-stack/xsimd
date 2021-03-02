/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <functional>

#include "test_utils.hpp"

template <class B>
class batch : public testing::Test
{
protected:

    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;

    array_type lhs;
    array_type rhs;

    batch()
    {
        init_operands();
    }

    void test_load_store() const
    {
        array_type res;
        batch_type b;
        b.load_unaligned(lhs.data());
        b.store_unaligned(res.data());
        EXPECT_EQ(res, lhs) << print_function_name("load_unaligned / store_unaligned");

        alignas(XSIMD_DEFAULT_ALIGNMENT) array_type arhs(this->rhs);
        alignas(XSIMD_DEFAULT_ALIGNMENT) array_type ares;
        b.load_aligned(arhs.data());
        b.store_aligned(ares.data());
        EXPECT_EQ(ares, rhs) << print_function_name("load_aligned / store_aligned");
    }

    void test_constructors() const
    {
        array_type tmp;
        std::fill(tmp.begin(), tmp.end(), value_type(2));
        batch_type b0(2);
        EXPECT_EQ(b0, tmp) << print_function_name("batch(value_type)");

        batch_type b1(lhs.data());
        EXPECT_EQ(b1, lhs) << print_function_name("batch(value_type*)");
    }

    void test_arithmetic() const
    {
        // batch + batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::plus<value_type>());
            batch_type res = batch_lhs() + batch_rhs();
            EXPECT_BATCH_EQ(res, expected) << print_function_name("batch + batch");
        }
    }

private:

    batch_type batch_lhs() const
    {
        return batch_type(lhs.data());
    }

    batch_type batch_rhs() const
    {
        return batch_type(rhs.data());
    }

    template <class T = value_type>
    xsimd::enable_integral_t<T, void> init_operands()
    {
        for (size_t i = 0; i < size; ++i)
        {
            bool negative_lhs = std::is_signed<T>::value && (i % 2 == 1);
            lhs[i] = value_type(i) * (negative_lhs ? -10 : 10);
            rhs[i] = value_type(i) + value_type(4);
        }
    }

    template <class T = value_type>
    xsimd::enable_floating_point_t<T, void> init_operands()
    {
        for (size_t i = 0; i < size; ++i)
        {
            lhs[i] = value_type(i) / 4 + value_type(1.2) * std::sqrt(value_type(i + 0.25));
            rhs[i] = value_type(10.2) / (i + 2) + value_type(0.25);
        }
    }
};

TYPED_TEST_SUITE_P(batch);

TYPED_TEST_P(batch, load_store)
{
    this->test_load_store();
}

TYPED_TEST_P(batch, constructors)
{
    this->test_constructors();
}

TYPED_TEST_P(batch, arithmetic)
{
    this->test_arithmetic();
}

REGISTER_TYPED_TEST_SUITE_P(
    batch,
    load_store,
    constructors,
    arithmetic
);


#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
INSTANTIATE_TYPED_TEST_SUITE_P(sse,
                               batch,
                               sse_types,
                               simd_test_names);
#endif

/***************************************************************************
 * Copyright (c) Toby Davis                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ELEMENT_REFERENCE_HPP
#define XSIMD_ELEMENT_REFERENCE_HPP

namespace xsimd
{
    template <class T, class A = default_arch>
    class batch;

    template <typename BatchType>
    class batch_element_reference
    {
    public:
        using Scalar = decltype(std::declval<BatchType>().get(0));

        batch_element_reference() = default;
        batch_element_reference(const batch_element_reference& other) = default;
        batch_element_reference(batch_element_reference&& other) = default;

        batch_element_reference(BatchType& data, uint64_t index)
            : m_data(data)
            , m_index(index)
        {
        }

        auto operator=(const batch_element_reference& other) -> batch_element_reference& = default;
        auto operator=(batch_element_reference&& other) -> batch_element_reference& = default;

        auto operator=(const Scalar& scalar) -> batch_element_reference&
        {
            set(scalar);
            return *this;
        }

        auto get() const
        {
            return m_data.get(m_index);
        }

        void set(const Scalar& other)
        {
            m_data.set(m_index, other);
        }

#define SIMD_REF_OP(OP_)                                              \
    auto operator OP_(const batch_element_reference& other) -> Scalar \
    {                                                                 \
        return get() OP_ other.get();                                   \
    }                                                                 \
                                                                      \
    auto operator OP_(const Scalar& other) -> Scalar                  \
    {                                                                 \
        return get() OP_ other;                                         \
    }

#define SIMD_REF_INPLACE_OP(OP_)                                                           \
    auto operator OP_##=(const batch_element_reference& other) -> batch_element_reference& \
    {                                                                                      \
        set(get() OP_ other.get());                                                        \
        return *this;                                                                      \
    }                                                                                      \
                                                                                           \
    auto operator OP_##=(const Scalar& other) -> batch_element_reference&                  \
    {                                                                                      \
        set(get() OP_ other);                                                              \
        return *this;                                                                      \
    }

        SIMD_REF_OP(+)
        SIMD_REF_OP(-)
        SIMD_REF_OP(*)
        SIMD_REF_OP(/)
        SIMD_REF_OP(&)
        SIMD_REF_OP(|)
        SIMD_REF_OP(^)
        SIMD_REF_OP(>)
        SIMD_REF_OP(<)
        SIMD_REF_OP(>=)
        SIMD_REF_OP(<=)
        SIMD_REF_OP(==)
        SIMD_REF_OP(!=)
        SIMD_REF_OP(&&)
        SIMD_REF_OP(||)
        SIMD_REF_OP(<<)
        SIMD_REF_OP(>>)

        SIMD_REF_INPLACE_OP(+)
        SIMD_REF_INPLACE_OP(-)
        SIMD_REF_INPLACE_OP(*)
        SIMD_REF_INPLACE_OP(/)
        SIMD_REF_INPLACE_OP(&)
        SIMD_REF_INPLACE_OP(|)
        SIMD_REF_INPLACE_OP(^)
        SIMD_REF_INPLACE_OP(<<)
        SIMD_REF_INPLACE_OP(>>)

#undef SIMD_REF_OP
#undef SIMD_REF_INPLACE_OP

        template <typename T>
        operator T() const
        {
            return static_cast<T>(get());
        }

    private:
        BatchType& m_data;
        uint64_t m_index;
    };

#define BATCH_ELEMENT_REF_UNARY_OP(NAME_)                          \
    template <typename T>                                          \
    inline auto NAME_(const batch_element_reference<T>& batch_ref) \
    {                                                              \
        return NAME_(batch_ref.get());                             \
    }

    BATCH_ELEMENT_REF_UNARY_OP(sin)
    BATCH_ELEMENT_REF_UNARY_OP(cos)
    BATCH_ELEMENT_REF_UNARY_OP(tan)
    BATCH_ELEMENT_REF_UNARY_OP(asin)
    BATCH_ELEMENT_REF_UNARY_OP(acos)
    BATCH_ELEMENT_REF_UNARY_OP(atan)
    BATCH_ELEMENT_REF_UNARY_OP(sinh)
    BATCH_ELEMENT_REF_UNARY_OP(cosh)
    BATCH_ELEMENT_REF_UNARY_OP(tanh)
    BATCH_ELEMENT_REF_UNARY_OP(asinh)
    BATCH_ELEMENT_REF_UNARY_OP(acosh)
    BATCH_ELEMENT_REF_UNARY_OP(atanh)
    BATCH_ELEMENT_REF_UNARY_OP(sqrt)
    BATCH_ELEMENT_REF_UNARY_OP(cbrt)
    BATCH_ELEMENT_REF_UNARY_OP(exp)
    BATCH_ELEMENT_REF_UNARY_OP(exp2)
    BATCH_ELEMENT_REF_UNARY_OP(exp10)
    BATCH_ELEMENT_REF_UNARY_OP(log)
    BATCH_ELEMENT_REF_UNARY_OP(log2)
    BATCH_ELEMENT_REF_UNARY_OP(log10)

#undef BATCH_ELEMENT_REF_UNARY_OP
};

template <typename BatchType>
auto operator<<(std::ostream& os, const xsimd::batch_element_reference<BatchType>& batch_ref) -> std::ostream&
{
    return os << batch_ref.get();
}

#endif // XSIMD_ELEMENT_REFERENCE_HPP

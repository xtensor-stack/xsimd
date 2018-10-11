/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille, Sylvain Corlay and   *
* Martin Renou                                                             *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_NEON_UINT32_HPP
#define XSIMD_NEON_UINT32_HPP

#include <utility>

#include "xsimd_base.hpp"
#include "xsimd_neon_bool.hpp"

namespace xsimd
{
    /**********************
     * batch<uint32_t, 4> *
     **********************/

    template <>
    struct simd_batch_traits<batch<uint32_t, 4>>
    {
        using value_type = uint32_t;
        static constexpr std::size_t size = 4;
        using batch_bool_type = batch_bool<uint32_t, 4>;
        static constexpr std::size_t align = XSIMD_DEFAULT_ALIGNMENT;
    };

    template <>
    class batch<uint32_t, 4> : public simd_batch<batch<uint32_t, 4>>
    {
    public:

        using simd_type = uint32x4_t;

        using base_type = simd_batch<batch<uint32_t, 4>>;

        batch();
        explicit batch(uint32_t d);

        template <class... Args, class Enable = detail::is_array_initializer_t<uint32_t, 4, Args...>>
        batch(Args... args);
        explicit batch(const uint32_t* src);

        batch(const uint32_t* src, aligned_mode);
        batch(const uint32_t* src, unaligned_mode);

        batch(const simd_type& rhs);
        batch& operator=(const simd_type& rhs);

        operator simd_type() const;

        XSIMD_DECLARE_LOAD_STORE_ALL(uint32_t, 4);

        using base_type::load_aligned;
        using base_type::load_unaligned;
        using base_type::store_aligned;
        using base_type::store_unaligned;

        uint32_t operator[](std::size_t index) const;

    private:

        simd_type m_value;
    };

    batch<uint32_t, 4> operator<<(const batch<uint32_t, 4>& lhs, uint32_t rhs);
    batch<uint32_t, 4> operator>>(const batch<uint32_t, 4>& lhs, uint32_t rhs);
}

#endif

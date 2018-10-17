/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille, Sylvain Corlay and   *
* Martin Renou                                                             *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_NEON_UINT64_HPP
#define XSIMD_NEON_UINT64_HPP

#include "xsimd_base.hpp"

namespace xsimd
{

    /**********************
     * batch<uint64_t, 2> *
     **********************/

    template <>
    struct simd_batch_traits<batch<uint64_t, 2>>
    {
        using value_type = uint64_t;
        static constexpr std::size_t size = 2;
        using batch_bool_type = batch_bool<uint64_t, 2>;
        static constexpr std::size_t align = XSIMD_DEFAULT_ALIGNMENT;
    };

    template <>
    class batch<uint64_t, 2> : public simd_batch<batch<uint64_t, 2>>
    {
    public:

        using self_type = batch<uint64_t, 2>;
        using base_type = simd_batch<self_type>;
        using simd_type = uint64x2_t;

        batch();
        explicit batch(uint64_t src);

        template <class... Args, class Enable = detail::is_array_initializer_t<uint64_t, 2, Args...>>
        batch(Args... args);
        explicit batch(const uint64_t* src);

        batch(const uint64_t* src, aligned_mode);
        batch(const uint64_t* src, unaligned_mode);

        batch(const simd_type& rhs);
        batch& operator=(const simd_type& rhs);

        operator simd_type() const;

        XSIMD_DECLARE_LOAD_STORE_ALL(uint64_t, 2);
        XSIMD_DECLARE_LOAD_STORE_LONG(uint64_t, 2);

        using base_type::load_aligned;
        using base_type::load_unaligned;
        using base_type::store_aligned;
        using base_type::store_unaligned;

        uint64_t operator[](std::size_t index) const;

    private:

        simd_type m_value;
    };

    batch<uint64_t, 2> operator<<(const batch<uint64_t, 2>& lhs, int64_t rhs);
    batch<uint64_t, 2> operator>>(const batch<uint64_t, 2>& lhs, int64_t rhs);
    batch<uint64_t, 2> operator<<(const batch<uint64_t, 2>& lhs, const batch<int64_t, 2>& rhs);

    /************************************
    * batch<uint64_t, 2> implementation *
    *************************************/

    inline batch<uint64_t, 2>::batch()
    {
    }

    inline batch<uint64_t, 2>::batch(uint64_t src)
        : m_value(vdupq_n_u64(src))
    {
    }

    template <class... Args, class>
    inline batch<uint64_t, 2>::batch(Args... args)
        : m_value{static_cast<uint64_t>(args)...}
    {
    }

    inline batch<uint64_t, 2>::batch(const uint64_t* src)
        : m_value(vld1q_u64(src))
    {
    }

    inline batch<uint64_t, 2>::batch(const uint64_t* src, aligned_mode)
        : batch(src)
    {
    }

    inline batch<uint64_t, 2>::batch(const uint64_t* src, unaligned_mode)
        : batch(src)
    {
    }

    inline batch<uint64_t, 2>::batch(const simd_type& rhs)
        : m_value(rhs)
    {
    }

    inline batch<uint64_t, 2>& batch<uint64_t, 2>::operator=(const simd_type& rhs)
    {
        m_value = rhs;
        return *this;
    }

    XSIMD_DEFINE_LOAD_STORE(uint64_t, 2, int8_t, XSIMD_DEFAULT_ALIGNMENT);
    XSIMD_DEFINE_LOAD_STORE(uint64_t, 2, uint8_t, XSIMD_DEFAULT_ALIGNMENT);
    XSIMD_DEFINE_LOAD_STORE(uint64_t, 2, int16_t, XSIMD_DEFAULT_ALIGNMENT);
    XSIMD_DEFINE_LOAD_STORE(uint64_t, 2, uint16_t, XSIMD_DEFAULT_ALIGNMENT);

    inline batch<uint64_t, 2>& batch<uint64_t, 2>::load_aligned(const int32_t* src)
    {
        int32x2_t tmp = vld1_s32(src);
        m_value = vreinterpretq_u64_s64(vmovl_s32(tmp));
        return *this;
    }

    inline batch<uint64_t, 2>& batch<uint64_t, 2>::load_unaligned(const int32_t* src)
    {
        return load_aligned(src);
    }

    inline batch<uint64_t, 2>& batch<uint64_t, 2>::load_aligned(const uint32_t* src)
    {
        uint32x2_t tmp = vld1_u32(src);
        m_value = vmovl_u32(tmp);
        return *this;
    }

    inline batch<uint64_t, 2>& batch<uint64_t, 2>::load_unaligned(const uint32_t* src)
    {
        return load_aligned(src);
    }

    inline batch<uint64_t, 2>& batch<uint64_t, 2>::load_aligned(const int64_t* src)
    {
        m_value = vreinterpretq_u64_s64(vld1q_s64(src));
        return *this;
    }

    inline batch<uint64_t, 2>& batch<uint64_t, 2>::load_unaligned(const int64_t* src)
    {
        return load_aligned(src);
    }

    inline batch<uint64_t, 2>& batch<uint64_t, 2>::load_aligned(const uint64_t* src)
    {
        m_value = vld1q_u64(src);
        return *this;
    }

    inline batch<uint64_t, 2>& batch<uint64_t, 2>::load_unaligned(const uint64_t* src)
    {
        return load_aligned(src);
    }

    XSIMD_DEFINE_LOAD_STORE_LONG(uint64_t, 2, 16)

    inline batch<uint64_t, 2>& batch<uint64_t, 2>::load_aligned(const float* src)
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        m_value = vcvtq_u64_f64(vcvt_f64_f32(vld1_f32(src)));
    #else
        m_value = uint64x2_t{
            static_cast<uint64_t>(src[0]),
            static_cast<uint64_t>(src[1])
        };
    #endif
        return *this;
    }

    inline batch<uint64_t, 2>& batch<uint64_t, 2>::load_unaligned(const float* src)
    {
        return load_unaligned(src);
    }

    inline batch<uint64_t, 2>& batch<uint64_t, 2>::load_aligned(const double* src)
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        m_value = vcvtq_u64_f64(vld1q_f64(src));
    #else
        m_value = uint64x2_t{
            static_cast<uint64_t>(src[0]),
            static_cast<uint64_t>(src[1])
        };
    #endif
        return *this;
    }

    inline batch<uint64_t, 2>& batch<uint64_t, 2>::load_unaligned(const double* src)
    {
        return load_aligned(src);
    }

    inline void batch<uint64_t, 2>::store_aligned(int32_t* dst) const
    {
        int32x2_t tmp = vmovn_s64(vreinterpretq_s64_u64(m_value));
        vst1_s32((int32_t*)dst, tmp);
    }

    inline void batch<uint64_t, 2>::store_unaligned(int32_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<uint64_t, 2>::store_aligned(uint32_t* dst) const
    {
        uint32x2_t tmp = vmovn_u64(m_value);
        vst1_u32((uint32_t*)dst, tmp);
    }

    inline void batch<uint64_t, 2>::store_unaligned(uint32_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<uint64_t, 2>::store_aligned(int64_t* dst) const
    {
        vst1q_s64(dst, vreinterpretq_s64_u64(m_value));
    }

    inline void batch<uint64_t, 2>::store_unaligned(int64_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<uint64_t, 2>::store_aligned(uint64_t* dst) const
    {
        vst1q_u64(dst, m_value);
    }

    inline void batch<uint64_t, 2>::store_unaligned(uint64_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<uint64_t, 2>::store_aligned(float* dst) const
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        vst1_f32(dst, vcvt_f32_f64(vcvtq_f64_u64(m_value)));
    #else
        dst[0] = static_cast<float>(m_value[0]);
        dst[1] = static_cast<float>(m_value[1]);
    #endif
    }

    inline void batch<uint64_t, 2>::store_unaligned(float* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<uint64_t, 2>::store_aligned(double* dst) const
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        vst1q_f64(dst, vcvtq_f64_u64(m_value));
    #else
        dst[0] = static_cast<double>(m_value[0]);
        dst[1] = static_cast<double>(m_value[1]);
    #endif
    }

    inline void batch<uint64_t, 2>::store_unaligned(double* dst) const
    {
        store_aligned(dst);
    }

    inline batch<uint64_t, 2>::operator simd_type() const
    {
        return m_value;
    }

    inline uint64_t batch<uint64_t, 2>::operator[](std::size_t index) const
    {
        return m_value[index];
    }

    namespace detail
    {
        template <>
        struct batch_kernel<uint64_t, 2>
        {
            using batch_type = batch<uint64_t, 2>;
            using value_type = uint64_t;
            using batch_bool_type = batch_bool<uint64_t, 2>;

            static batch_type neg(const batch_type& rhs)
            {
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
                return vreinterpretq_u64_s64(vnegq_s64(vreinterpretq_s64_u64(rhs)));
#else
                return batch<uint64_t, 2>(-rhs[0], -rhs[1]);
#endif
            }

            static batch_type add(const batch_type& lhs, const batch_type& rhs)
            {
                return vaddq_u64(lhs, rhs);
            }

            static batch_type sub(const batch_type& lhs, const batch_type& rhs)
            {
                return vsubq_u64(lhs, rhs);
            }

            static batch_type mul(const batch_type& lhs, const batch_type& rhs)
            {
                return { lhs[0] * rhs[0], lhs[1] * rhs[1] };
            }

            static batch_type div(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION && defined(XSIMD_FAST_INTEGER_DIVISION)
                return vcvtq_u64_f64(vcvtq_f64_u64(lhs) / vcvtq_f64_u64(rhs));
#else
                return{ lhs[0] / rhs[0], lhs[1] / rhs[1] };
#endif
            }

            static batch_type mod(const batch_type& lhs, const batch_type& rhs)
            {
                return{ lhs[0] % rhs[0], lhs[1] % rhs[1] };
            }

            static batch_bool_type eq(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
                return vceqq_u64(lhs, rhs);
#else
                return batch_bool<uint64_t, 2>(lhs[0] == rhs[0], lhs[1] == rhs[1]);
#endif
            }

            static batch_bool_type neq(const batch_type& lhs, const batch_type& rhs)
            {
                return !(lhs == rhs);
            }

            static batch_bool_type lt(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
                return vcltq_u64(lhs, rhs);
#else
                return batch_bool<uint64_t, 2>(lhs[0] < rhs[0], lhs[1] < rhs[1]);
#endif
            }

            static batch_bool_type lte(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
                return vcleq_u64(lhs, rhs);
#else
                return batch_bool<uint64_t, 2>(lhs[0] <= rhs[0], lhs[1] <= rhs[1]);
#endif
            }

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return vandq_u64(lhs, rhs);
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return vorrq_u64(lhs, rhs);
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return veorq_u64(lhs, rhs);
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(rhs)));
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return vbicq_u64(lhs, rhs);
            }

            static batch_type min(const batch_type& lhs, const batch_type& rhs)
            {
                return { lhs[0] < rhs[0] ? lhs[0] : rhs[0],
                         lhs[1] < rhs[1] ? lhs[1] : rhs[1] };
            }

            static batch_type max(const batch_type& lhs, const batch_type& rhs)
            {
                return { lhs[0] > rhs[0] ? lhs[0] : rhs[0],
                         lhs[1] > rhs[1] ? lhs[1] : rhs[1] };
            }

            static batch_type abs(const batch_type& rhs)
            {
                return rhs;
            }

            static batch_type fma(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return x * y + z;
            }

            static batch_type fms(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return x * y - z;
            }

            static batch_type fnma(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return -x * y + z;
            }

            static batch_type fnms(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return -x * y - z;
            }

            static value_type hadd(const batch_type& rhs)
            {
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
                return vaddvq_u64(rhs);
#else
                return rhs[0] + rhs[1];
#endif
            }

            static batch_type select(const batch_bool_type& cond, const batch_type& a, const batch_type& b)
            {
                return vbslq_u64(cond, a, b);
            }
        };

        inline batch<uint64_t, 2> shift_left(const batch<uint64_t, 2>& lhs, const int n)
        {
            switch(n)
            {
                case 0: return lhs;
                XSIMD_REPEAT_64(vshlq_n_u64);
                default: break;
            }
            return batch<uint64_t, 2>(uint64_t(0));
        }

        inline batch<uint64_t, 2> shift_right(const batch<uint64_t, 2>& lhs, const int n)
        {
            switch(n)
            {
                case 0: return lhs;
                XSIMD_REPEAT_64(vshrq_n_u64);
                default: break;
            }
            return batch<uint64_t, 2>(uint64_t(0));
        }
    }

    inline batch<uint64_t, 2> operator<<(const batch<uint64_t, 2>& lhs, int64_t rhs)
    {
        return detail::shift_left(lhs, rhs);
    }

    inline batch<uint64_t, 2> operator>>(const batch<uint64_t, 2>& lhs, int64_t rhs)
    {
        return detail::shift_right(lhs, rhs);
    }

    inline batch<uint64_t, 2> operator<<(const batch<uint64_t, 2>& lhs, const batch<int64_t, 2>& rhs)
    {
        return vshlq_u64(lhs, rhs);
    }
}

#endif

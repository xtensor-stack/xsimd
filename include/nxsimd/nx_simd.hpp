//
// Copyright (c) 2016 Johan Mabille
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
//

#ifndef NX_SIMD_HPP
#define NX_SIMD_HPP

#include "types/nx_simd_traits.hpp"

namespace nxsimd
{


    // Data transfer instructions
    
    template <class T>
    simd_type<T> set_simd(const T& value);

    template <class T>
    simd_type<T> load_aligned(const T* src);

    template <class T>
    void load_aligned(const T* src, simd_type<T>& dst);

    template <class T>
    simd_type<T> load_unaligned(const T* src);

    template <class T>
    void load_unaligned(const T* src, simd_type<T>& dst);

    template <class T>
    void store_aligned(T* dst, const simd_type<T>& src);

    template <class T>
    void store_unaligned(T* dst, const simd_type<T>& src);

    // Load / store generic functions

    struct aligned_mode {}
    struct unaligned_mode {}

    template <class T>
    simd_type<T> load_simd(const T* src, aligned_mode);

    template <class T>
    void load_simd(const T* src, simd_type<T>& dst, aligned_mode);

    template <class T>
    simd_type<T> load_simd(const T* src, unaligned_mode);

    template <class T>
    void load_simd(const T* src, simd_type<T>& dst, unaligned_mode);

    template <class T>
    void store_simd(T* dst, const simd_type<T>& src, aligned_mode);

    template <class T>
    void store_simd(T* dst, const simd_type<T>& src, unaligned_mode);
    
    // Prefetch
    
    template <class T>
    void prefetch(const T* address);


    /***************************
     * detail implementation 
     ***************************/

    namespace detail
    {
        // Common implementation of SIMD functions for types supported
        // by vectorization.
        template <class T, class V>
        struct simd_function_invoker
        {
            inline static V set_simd(const T& value)
            {
                return V(value);
            }
        
            inline static V load_aligned(const T* src)
            {
                V res;
                return res.load_aligned(src);
            }

            inline static void load_aligned(const T* src, V& dst)
            {
                dst.load_aligned(src);
            }

            inline static V load_unaligned(const T* src)
            {
                V res;
                return res.load_unaligned(src);
            }

            inline static void load_unaligned(const T* src, V& dst)
            {
                return dst.load_unaligned(src);
            }

            inline static void store_aligned(T* dst, const V& src)
            {
                src.store_aligned(dst);
            }

            inline static void store_unaligned(T* dst, const V& src)
            {
                src.store_unaligned(dst);
            }
        };

        // Default implementation of SIMD functions for types not supported
        // by vectorization.
        template <class T>
        struct simd_function_invoker<T, T>
        {
            inline static T set_simd(const T& value)
            {
                return value;
            }

            inline static T load_aligned(const T* src)
            {
                return *src;
            }

            inline static void load_aligned(const T* src, V& dst)
            {
                dst = *src;
            }

            inline static T load_unaligned(const T* src)
            {
                return *src;
            }

            inline static void load_unaligned(const T* src, V& dst)
            {
                dst = *src;
            }

            inline static void store_aligned(T* dst, const V& src)
            {
                *dst = src;
            }

            inline static void store_unaligned(T* dst, const V& src)
            {
                *dst = src;
            }
        };
    }


    /***********************************************
     * Data transfer instructions implementation
     ***********************************************/
    
    template <class T>
    inline simd_type<T> set_simd(const T& value)
    {
        return detail::simd_function_invoker<T, simd_type<T>>::set_simd(value);
    }

    template <class T>
    inline simd_type<T> load_aligned(const T* src)
    {
        return detail::simd_function_invoker<T, simd_type<T>>::load_aligned(src);
    }

    template <class T>
    inline void load_aligned(const T* src, simd_type<T>& dst)
    {
        detail::simd_function_invoker<T, simd_type<T>>::load_aligned(src, dst);
    }

    template <class T>
    inline simd_type<T> load_unaligned(const T* src)
    {
        return detail::simd_function_invoker<T, simd_type<T>>::load_unaligned(src);
    }

    template <class T>
    inline void load_unaligned(const T* src, simd_type<T>& dst)
    {
        detail::simd_function_invoker<T, simd_type<T>>::load_unaligned(src, dst);
    }

    template <class T>
    inline void store_aligned(T* dst, const simd_type<T>& src)
    {
        detail::simd_function_invoker<T, simd_type<T>>::store_aligned(dst, src);
    }

    template <class T>
    inline void store_unaligned(T* dst, const simd_type<T>& src)
    {
        detail::simd_function_invoker<T, simd_type<T>>::store_unaligned(dst, src);
    }


    /***************************************************
     * Load / store generic functions implementation
     ***************************************************/

    template <class T>
    inline simd_type<T> load_simd(const T* src, aligned_mode)
    {
        return load_aligned(src);
    }

    template <class T>
    inline void load_simd(const T* src, simd_type<T>& dst, aligned_mode)
    {
        load_aligned(src, dst);
    }

    template <class T>
    inline simd_type<T> load_simd(const T* src, unaligned_mode)
    {
        return load_unaligned(src);
    }

    template <class T>
    inline void load_simd(const T* src, simd_type<T>& dst, unaligned_mode)
    {
        load_unaligned(src, dst);
    }

    template <class T>
    inline void store_simd(T* dst, const simd_type<T>& src, aligned_mode)
    {
        store_aligned(dst, src);
    }

    template <class T>
    inline void store_simd(T* dst, const simd_type<T>& src, unaligned_mode)
    {
        store_unaligned(dst, src);
    }
    

    /*****************************
     * Prefetch implementation
     *****************************/
    
    template <class T>
    inline void prefetch(const T* address)
    {
    }

#if defined(NX_USE_SSE) || defined(NX_USE_AVX)

    template <>
    inline void prefetch<int>(const int* address)
    {
        _mm_prefetch(reinterpret_cast<const char*>(address), _MM_HINT_T0);
    }

    template <>
    inline void prefetch<float>(const float* address)
    {
        _mm_prefetch(reinterpret_cast<const char*>(address), _MM_HINT_T0);
    }

    template <>
    inline void prefetch<double>(const double* address)
    {
        _mm_prefetch(reinterpret_cast<const char*>(address), _MM_HINT_T0);
    }

#endif

}

#endif


//
// Copyright (c) 2016 Johan Mabille
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
//

#ifndef NX_SIMD_HPP
#define NX_SIMD_HPP

#include "nx_simd_traits.hpp"

namespace nxsimd
{


    // Data transfer instructions
    
    template <class T>
    simd_type<T> set1_p(const T& value);

    template <class T>
    simd_type<T> load_p(const T* src);

    template <class T>
    void load_p(const T* src, simd_type<T>& dst);

    template <class T>
    simd_type<T> loadu_p(const T* src);

    template <class T>
    void loadu_p(const T* src, simd_type<T>& dst);

    template <class T>
    void store_p(T* dst, const simd_type<T>& src);

    template <class T>
    void storeu_p(T* dst, const simd_type<T>& src);

    // Load / store generic functions

    struct aligned_mode {}
    struct unaligned_mode {}

    template <class T>
    simd_type<T> loadg_p(const T* src, aligned_mode);

    template <class T>
    void load_p(const T* src, simd_type<T>& dst, aligned_mode);

    template <class T>
    simd_type<T> loadg_p(const T* src, unaligned_mode);

    template <class T>
    void loadg_p(const T* src, simd_type<T>& dst, unaligned_mode);

    template <class T>
    void storeg_p(T* dst, const simd_type<T>& src, aligned_mode);

    template <class T>
    void storeg_p(T* dst, const simd_type<T>& src, unaligned_mode);
    
    // Prefetch
    
    template <class T>
    void prefetch(const T* address);


    namespace detail
    {
        // Common implementation of SIMD functions for types supported
        // by vectorization.
        template <class T, class V>
        struct simd_function_invoker
        {
            static V set1_p(const T& value);
        
            static V load_p(const T* src);
            static void load_p(const T* src, V& dst);

            static V loadu_p(const T* src);
            static void loadu_p(const T* src, V& dst);

            static void store_p(T* dst, const V& src);
            static void storeu_p(T* dst, const V& src);
        };

        // Default implementation of SIMD functions for types not supported
        // by vectorization.
        template <class T>
        struct simd_function_invoker<T, T>
        {
            static T set1_p(const T& value);

            static T load_p(const T* src);
            static void load_p(const T* src, V& dst);

            static T loadu_p(const T* src);
            static void loadu_p(const T* src, V& dst);

            static void store_p(T* dst, const V& src);
            static void storeu_p(T* dst, const V& src);
        };
    }

}

#endif


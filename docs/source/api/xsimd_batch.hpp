/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

// This file is for generating the documentation

namespace xsimd
{

    /**
     * @class batch_bool
     * @brief Batch of boolean values.
     *
     * The batch_bool class represents a batch of boolean values, that can be used
     * in operations involving batches of integer or floating point values. The boolean
     * values are stored as integer or floating point values, depending on the type of
     * batch they are dedicated to.
     *
     * @tparam T The value type used for encoding boolean.
     * @tparam N The number of scalar in the batch.
     */
    template <class T, std::size_t N>
    class batch_bool : public simd_batch_bool<batch_bool<T, N>>
    {

    public:

        /**
         * Builds an uninitialized batch of boolean values.
         */
        batch_bool();

        /**
         * Initializes all the values of the batch to \c b.
         */
        explicit batch_bool(bool b);

        /**
         * Initializes a batch of booleans with the specified boolean values.
         */
        batch_bool(bool b0, ..., bool bn);

        /**
         * Initializes a batch of boolean with the specified SIMD value.
         */
        batch_bool(const simd_data& rhs);

        /**
         * Assigns the specified SIMD value.
         */
        batch_bool& operator=(const simd_data& rhs);

        /**
         * Converts \c this to a SIMD value.
         */
        operator simd_data() const;
    };

    template <class T, std::size_t N>
    class batch : public simd_batch<batch<T, N>>
    {

    public:

        /**
         * Builds an uninitialized batch.
         */
        batch();

        /**
         * Initializes all the values of the batch to \c b.
         */
        explicit batch(T f);

        /**
         * Initializes a batch with the specified scalar values.
         */
        batch(T f0, ..., T f3);

        /*
         * Initializes a batch to the values pointed by \c src; \c src
         * does not need to be aligned.
         */
        explicit batch(const T* src);

        /**
         * Initializes a batch to the N contiguous values pointed by \c src; \c src
         * is not required to be aligned.
         */
        batch(const T* src, aligned_mode);

        /**
         * Initializes a batch to the values pointed by \c src; \c src
         * must be aligned.
         */
        batch(const T* src, unaligned_mode);

        /**
         * Initializes a batch with the specified SIMD value.
         */
        batch(const simd_data& rhs);

        /**
         * Assigns the specified SIMD value to the batch.
         */
        batch& operator=(const simd_data& rhs);

        /**
         * Converts \c this to a SIMD value.
         */
        operator simd_data() const;

        /**
         * Loads the N contiguous values pointed by \c src into the batch.
         * \c src must be aligned.
         */
        batch& load_aligned(const T* src);

        /**
         * Loads the N contiguous values pointed by \c src into the batch.
         * \c src is not required to be aligned.
         */
        batch& load_unaligned(const T* src);

        /**
         * Stores the N values of the batch into a contiguous array
         * pointed by \c dst. \c dst must be aligned.
         */
        void store_aligned(T* dst) const;

        /**
         * Stores the N values of the batch into a contiguous array
         * pointed by \c dst. \c dst is not required to be aligned.
         */
        void store_unaligned(T* dst) const;

        /**
         * Return the i-th scalar in the batch.
         */
        T operator[](std::size_t i) const;
    };
}

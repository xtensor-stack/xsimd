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

    /**
     * @defgroup batch_bool_logical Logical operators
     */

    /**
     * @ingroup batch_bool_logical
     *
     * Computes the bitwise and of the batches \c lhs and \c rhs.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise and.
     */
    template <class T, std::size_t N>
    batch_bool<T, N> operator&(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs);

    /**
     * @ingroup batch_bool_logical
     *
     * Computes the bitwise or of the batches \c lhs and \c rhs.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise or.
     */
    template <class T, std::size_t N>
    batch_bool<T, N> operator|(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs);

    /**
     * @ingroup batch_bool_logical
     *
     * Computes the bitwise xor of the batches \c lhs and \c rhs.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise xor.
     */
    template <class T, std::size_t N>
    batch_bool<T, N> operator^(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs);

    /**
     * @ingroup batch_bool_logical
     *
     * Computes the bitwise not of the batch \c rhs.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise not.
     */
    template <class T, std::size_t N>
    batch_bool<T, N> operator~(const batch_bool<T, N>& rhs);

    /**
     * @ingroup batch_bool_logical
     *
     * Computes the bitwise and not of the batches \c lhs and \c rhs. Equivalent
     * to \verbatim lhs & ~rhs \endverbatim.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise not.
     */
    template <class T, std::size_t N>
    batch_bool<T, N> bitwise_andnot(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs);
 
    /**
     * @defgroup batch_bool_comparison Comparison operators
     */

    /**
     * @ingroup batch_bool_comparison
     *
     * Element-wise equality of batches \c lhs and \c rhs.
     * @param lhs batch involved in the comparison.
     * @param rhs batch involved in the comparison.
     * @return the result of the equality comparison.
     */
    template <class T, std::size_t N>
    batch_bool<T, N> operator==(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs);

    /**
     * @ingroup batch_bool_comparison
     *
     * Element-wise inequality of batches \c lhs and \c rhs.
     * @param lhs batch involved in the comparison.
     * @param rhs batch involved in the comparison.
     * @return the result of the inequality comparison.
     */
    template <class T, std::size_t N>
    batch_bool<T, N> operator!=(const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs);

    /**
     * @defgroup batch_bool_reducers Reducers
     */

    /**
     * @ingroup batch_bool_reducers
     *
     * Returns true if all the boolean values in the batch are true,
     * false otherwise.
     * @param rhs the batch to reduce.
     * @return a boolean scalar.
     */
    template <class T, std::size_t N>
    bool all(const batch_bool<T, N>& rhs);

    /**
     * @ingroup batch_bool_reducers
     *
     * Return true if any of the boolean values in the batch is true,
     * false otherwise.
     * @param rhs the batch to reduce.
     * @return a boolean scalar.
     */
    template <class T, std::size_t N>
    bool any(const batch_bool<T, N>& rhs);

    /**
     * @class batch
     * @brief Batch of integer or floating point values.
     *
     * The batch class represents a batch of integer or floating point values.
     * Types supported are int32_t, int64_t, float and double.
     *
     * @tparam T The value type.
     * @tparam N The number of scalar in the batch.
     */

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
         * Initializes a batch with the specified boolean values.
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

    /**
     * @defgroup batch_arithmetic Arithmetic operators
     */

    /**
     * @ingroup batch_arithmetic
     *
     * Computes the opposite of the batch \c rhs.
     * @param rhs batch involved in the operation.
     * @return the opposite of \c rhs.
     */
    template <class T, std::size_t N>
    batch<T, N> operator-(const batch<T, N>& rhs);
    
    /**
     * @ingroup batch_arithmetic
     *
     * Computes the sum of the batches \c lhs and \c rhs.
     * @param lhs batch involved in the addition.
     * @param rhs batch involved in the addition.
     * @return the result of the addition.
     */
    template <class T, std::size_t N>
    batch<T, N> operator+(const batch<T, N>& lhs, const batch<T, N>& rhs);
    
    /**
     * @ingroup batch_arithmetic
     *
     * Computes the difference of the batches \c lhs and \c rhs.
     * @param lhs batch involved in the difference.
     * @param rhs batch involved in the difference.
     * @return the result of the difference.
     */
    template <class T, std::size_t N>
    batch<T, N> operator-(const batch<T, N>& lhs, const batch<T, N>& rhs);
    
    /**
     * @ingroup batch_arithmetic
     *
     * Computes the product of the batches \c lhs and \c rhs.
     * @param lhs batch involved in the product.
     * @param rhs batch involved in the product.
     * @return the result of the product.
     */
    template <class T, std::size_t N>
    batch<T, N> operator*(const batch<T, N>& lhs, const batch<T, N>& rhs);

    /**
     * @ingroup batch_arithmetic
     *
     * Computes the division of the batch \c lhs by the batch \c rhs.
     * @param lhs batch involved in the division.
     * @param rhs batch involved in the division.
     * @return the result of the division.
     */
    template <class T, std::size_t N>
    batch<T, N> operator/(const batch<T, N>& lhs, const batch<T, N>& rhs);
    
    /**
     * @defgroup batch_comparison Comparison operators
     */

    /**
     * @ingroup batch_comparison
     *
     * Element-wise equality comparison of batches \c lhs and \c rhs.
     * @param lhs batch involved in the comparison.
     * @param rhs batch involved in the comparison.
     * @return a boolean batch.
     */
    template <class T, std::size_t N>
    batch_bool<T, 4> operator==(const batch<T, N>& lhs, const batch<T, N>& rhs);

    /**
     * @ingroup batch_comparison
     *
     * Element-wise inequality comparison of batches \c lhs and \c rhs.
     * @param lhs batch involved in the comparison.
     * @param rhs batch involved in the comparison.
     * @return a boolean batch.
     */
    template <class T, std::size_t N>
    batch_bool<T, 4> operator!=(const batch<T, N>& lhs, const batch<T, N>& rhs);

    /**
     * @ingroup batch_comparison
     *
     * Element-wise lesser than comparison of batches \c lhs and \c rhs.
     * @param lhs batch involved in the comparison.
     * @param rhs batch involved in the comparison.
     * @return a boolean batch.
     */
    template <class T, std::size_t N>
    batch_bool<T, 4> operator<(const batch<T, N>& lhs, const batch<T, N>& rhs);

    /**
     * @ingroup batch_comparison
     *
     * Element-wise lesser or equal to comparison of batches \c lhs and \c rhs.
     * @param lhs batch involved in the comparison.
     * @param rhs batch involved in the comparison.
     * @return a boolean batch.
     */
    template <class T, std::size_t N>
    batch_bool<T, 4> operator<=(const batch<T, N>& lhs, const batch<T, N>& rhs);

    /**
     * @defgroup batch_logical Logical operators
     */
    /**
     * @ingroup batch_logical
     *
     * Computes the bitwise and of the batches \c lhs and \c rhs.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise and.
     */
    template <class T, std::size_t N>
    batch<T, N> operator&(const batch<T, N>& lhs, const batch<T, N>& rhs);

    /**
     * @ingroup batch_logical
     *
     * Computes the bitwise or of the batches \c lhs and \c rhs.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise or.
     */
    template <class T, std::size_t N>
    batch<T, N> operator|(const batch<T, N>& lhs, const batch<T, N>& rhs);

    /**
     * @ingroup batch_logical
     *
     * Computes the bitwise xor of the batches \c lhs and \c rhs.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise xor.
     */
    template <class T, std::size_t N>
    batch<T, N> operator^(const batch<T, N>& lhs, const batch<T, N>& rhs);

    /**
     * @ingroup batch_logical
     *
     * Computes the bitwise not of the batches \c lhs and \c rhs.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise not.
     */
    template <class T, std::size_t N>
    batch<T, N> operator~(const batch<T, N>& rhs);

    /**
     * @ingroup batch_logical
     *
     * Computes the bitwise andnot of the batches \c lhs and \c rhs.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise andnot.
     */
    template <class T, std::size_t N>
    batch<T, N> bitwise_andnot(const batch<T, N>& lhs, const batch<T, N>& rhs);

    /**
     * Returns the smaller values of the batches \c x and \c y.
     * @param x a batch of integer or floating point values.
     * @param y a batch of integer or floating point values.
     * @return a batch of the smaller values.
     */
    template <class T, std::size_t N>
    batch<T, N> min(const batch<T, N>& x, const batch<T, N>& y);

    /**
     * Returns the larger values of the batches \c x and \c y.
     * @param x a batch of integer or floating point values.
     * @param y a batch of integer or floating point values.
     * @return a batch of the larger values.
     */
    template <class T, std::size_t N>
    batch<T, N> max(const batch<T, N>& x, const batch<T, N>& y);

    /**
     * Returns the smaller values of the batches \c x and \c y.
     * @param x a batch of floating point values.
      * @param y a batch of floating point values.
     * @return a batch of the smaller values.
     */
    template <class T, std::size_t N>
    batch<T, N> fmin(const batch<T, N>& x, const batch<T, N>& y);

    /**
     * Returns the larger values of the batches \c x and \c y.
     * @param x a batch of floating point values.
     * @param y a batch of floating point values.
     * @return a batch of the larger values.
     */
    template <class T, std::size_t N>
    batch<T, N> fmax(const batch<T, N>& x, const batch<T, N>& y);

    /**
     * Computes the absolute values of each scalar in the batch \c x.
     * @param x batch of integer or floating point values.
     * @return the asbolute values of \c x.
     */
    template <class T, std::size_t N>
    batch<T, N> abs(const batch<T, N>& x);

    /**
    * Computes the absolute values of each scalar in the batch \c x.
    * @param x batch floating point values.
    * @return the asbolute values of \c x.
    */
    template <class T, std::size_t N>
    batch<T, N> fabs(const batch<T, N>& x);

    /**
     * Computes the square root of the batch \c x.
     * @param x batch of floating point values.
     * @return the square root of \c x.
     */
    template <class T, std::size_t N>
    batch<T, N> sqrt(const batch<T, N>& x);

    /**
     * Computes <tt>(x*y) + z</tt> in a single instruction when possible.
     * @param x a batch of integer or floating point values.
     * @param y a batch of integer or floating point values.
     * @param z a batch of integer or floating point values.
     * @return the result of the fused multiply-add operation.
     */
    template <class T, std::size_t N>
    batch<T, N> fma(const batch<T, N>& x, const batch<T, N>& y, const batch<T, N>& z);

    /**
     * Computes <tt>(x*y) - z</tt> in a single instruction when possible.
     * @param x a batch of integer or floating point values.
     * @param y a batch of integer or floating point values.
     * @param z a batch of integer or floating point values.
     * @return the result of the fused multiply-sub operation.
     */
    template <class T, std::size_t N>
    batch<T, N> fms(const batch<T, N>& x, const batch<T, N>& y, const batch<T, N>& z);

    /**
     * Computes <tt>-(x*y) + z</tt> in a single instruction when possible.
     * @param x a batch of integer or floating point values.
     * @param y a batch of integer or floating point values.
     * @param z a batch of integer or floating point values.
     * @return the result of the fused negated multiply-add operation.
     */
    template <class T, std::size_t N>
    batch<T, N> fnma(const batch<T, N>& x, const batch<T, N>& y, const batch<T, N>& z);

    /**
     * Computes <tt>-(x*y) - z</tt> in a single instruction when possible.
     * @param x a batch of integer or floating point values.
     * @param y a batch of integer or floating point values.
     * @param z a batch of integer or floating point values.
     * @return the result of the fused negated multiply-sub operation.
     */
    template <class T, std::size_t N>
    batch<T, N> fnms(const batch<T, N>& x, const batch<T, N>& y, const batch<T, N>& z);

    /**
     * @defgroup batch_reducers Reducers
     */

    /**
     * @ingroup batch_reducers
     *
     * Adds all the scalars of the batch \c rhs. 
     * @param rhs batch involved in the reduction
     * @return the result of the reduction.
     */
    template <class T, std::size_t N>
    T hadd(const batch<T, N>& rhs);

    /**
     * @ingroup batch_reducers
     *
     * Parallel horizontal addition: adds the scalars of each batch
     * in the array pointed by \c row and store them in a returned
     * batch.
     * @param row an array of \c N batches
     * @return the result of the reduction.
     */
    template <class T, std::size_t N>
    batch<T, N> haddp(const batch<T, N>* row);

    /**
     * @defgroup batch_miscellaneous Miscellaneous
     */

    /**
     * @ingroup batch_miscellaneous
     *
     * Ternary operator for batches: selects values from the batches \c a or \c b
     * depending on the boolean values in \c cond. Equivalent to
     * \code{.cpp}
     * for(std::size_t i = 0; i < N; ++i)
     *     res[i] = cond[i] ? a[i] : b[i];
     * \endcode
     * @param cond batch condition.
     * @param a batch values for truthy condition.
     * @param b batch value for falsy condition.
     * @return the result of the selection.
     */
    template <class T, std::size_t N>
    batch<T, N> select(const batch_bool<T, N>& cond, const batch<T, N>& a, const batch<T, N>& b);

    /**
     * Determines if the scalars in the given batch \c x are NaN values.
     * @param x batch of floating point values.
     * @return a batch of booleans.
     */
    template <class T, std::size_t N>
    batch_bool<T, 4> isnan(const batch<T, N>& x);
}

/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_BASE_HPP
#define XSIMD_BASE_HPP

#include <cstddef>
#include <ostream>

#include "../memory/xsimd_alignment.hpp"

namespace xsimd
{

    template <class T, std::size_t N>
    class batch_bool;

    template <class T, std::size_t N>
    class batch;

    /*******************
     * simd_batch_bool *
     *******************/

    /**
     * @class simd_batch_bool
     * @brief Base class for batch of boolean values.
     *
     * The simd_batch_bool class is the base class for all classes representing
     * a batch of boolean values. Batch of boolean values is meant for operations
     * that may involve batches of integer or floating point values. Thus,
     * the boolean values are stored as integer or floating point values, and each
     * type of batch has its dedicated type of boolean batch.
     *
     * @tparam X The derived type
     * @sa simd_batch
     */
    template <class X>
    class simd_batch_bool
    {
    public:

        X& operator&=(const X& rhs);
        X& operator|=(const X& rhs);
        X& operator^=(const X& rhs);

        X& operator()();
        const X& operator()() const;

    protected:

        simd_batch_bool() = default;
        ~simd_batch_bool() = default;

        simd_batch_bool(const simd_batch_bool&) = default;
        simd_batch_bool& operator=(const simd_batch_bool&) = default;

        simd_batch_bool(simd_batch_bool&&) = default;
        simd_batch_bool& operator=(simd_batch_bool&&) = default;
    };

    template <class X>
    X operator&&(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>& rhs);

    template <class X>
    X operator&&(const simd_batch_bool<X>& lhs, bool rhs);

    template <class X>
    X operator&&(bool lhs, const simd_batch_bool<X>& rhs);

    template <class X>
    X operator||(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>& rhs);

    template <class X>
    X operator||(const simd_batch_bool<X>& lhs, bool rhs);

    template <class X>
    X operator||(bool lhs, const simd_batch_bool<X>& rhs);

    template <class X>
    X operator!(const simd_batch_bool<X>& rhs);

    /**************
     * simd_batch *
     **************/

    template <class X>
    struct simd_batch_traits;

    /**
     * @class simd_batch
     * @brief Base class for batch of integer or floating point values.
     *
     * The simd_batch class is the base class for all classes representing
     * a batch of integer or floating point values. Each type of batch (i.e.
     * a class inheriting from simd_batch) has its dedicated type of boolean
     * batch (i.e. a class inheriting from simd_batch_bool) for logical operations.
     *
     * @tparam X The derived type
     * @sa simd_batch_bool
     */
    template <class X>
    class simd_batch
    {
    public:

        using value_type = typename simd_batch_traits<X>::value_type;
        static std::size_t constexpr size = simd_batch_traits<X>::size;

        X& operator+=(const X& rhs);
        X& operator+=(const value_type& rhs);

        X& operator-=(const X& rhs);
        X& operator-=(const value_type& rhs);

        X& operator*=(const X& rhs);
        X& operator*=(const value_type& rhs);

        X& operator/=(const X& rhs);
        X& operator/=(const value_type& rhs);

        X& operator&=(const X& rhs);
        X& operator|=(const X& rhs);
        X& operator^=(const X& rhs);

        X& operator++();
        X& operator++(int);

        X& operator--();
        X& operator--(int);

        X& operator()();
        const X& operator()() const;

    protected:

        simd_batch() = default;
        ~simd_batch() = default;

        simd_batch(const simd_batch&) = default;
        simd_batch& operator=(const simd_batch&) = default;

        simd_batch(simd_batch&&) = default;
        simd_batch& operator=(simd_batch&&) = default;
    };

    template <class X>
    X operator+(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs);

    template <class X>
    X operator+(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator-(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs);

    template <class X>
    X operator-(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator*(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs);

    template <class X>
    X operator*(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator/(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs);

    template <class X>
    X operator/(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs);

    template <class X>
    typename simd_batch_traits<X>::batch_bool_type
    operator>(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    typename simd_batch_traits<X>::batch_bool_type
    operator>=(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    typename simd_batch_traits<X>::batch_bool_type
    operator!(const simd_batch<X>& rhs);

    template <class X>
    std::ostream& operator<<(std::ostream& out, const simd_batch<X>& rhs);

    /***************************
     * generic batch operators *
     ***************************/

    template <class T, std::size_t N>
    batch<T, N> operator&&(const batch<T, N>& lhs, const batch<T, N>& rhs);

    template <class T, std::size_t N>
    batch<T, N> operator||(const batch<T, N>& lhs, const batch<T, N>& rhs);

    template <class T, std::size_t N>
    batch<T, N> operator<<(const batch<T, N>& lhs, const batch<T, N>& rhs);

    template <class T, std::size_t N>
    batch<T, N> operator>>(const batch<T, N>& lhs, const batch<T, N>& rhs);

    /**********************************
     * simd_batch_bool implementation *
     **********************************/

    /**
     * @name Bitwise computed assignement
     */
    //@{
    /**
     * Assigns the bitwise and of \c rhs and \c this.
     * @param rhs the batch involved in the operation.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch_bool<X>::operator&=(const X& rhs)
    {
        (*this)() = (*this)() & rhs;
        return (*this)();
    }

    /**
     * Assigns the bitwise or of \c rhs and \c this.
     * @param rhs the batch involved in the operation.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch_bool<X>::operator|=(const X& rhs)
    {
        (*this)() = (*this)() | rhs;
        return (*this)();
    }

    /**
     * Assigns the bitwise xor of \c rhs and \c this.
     * @param rhs the batch involved in the operation.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch_bool<X>::operator^=(const X& rhs)
    {
        (*this)() = (*this)() ^ rhs;
        return (*this)();
    }
    //@}

    /**
     * @name Static downcast functions
     */
    //@{
    /**
     * Returns a reference to the actual derived type of the simd_batch_bool.
     */
    template <class X>
    inline X& simd_batch_bool<X>::operator()()
    {
        return *static_cast<X*>(this);
    }

    /**
     * Returns a constant reference to the actual derived type of the simd_batch_bool.
     */
    template <class X>
    const X& simd_batch_bool<X>::operator()() const
    {
        return *static_cast<const X*>(this);
    }
    //@}

    /**
     * @defgroup simd_batch_bool_logical Logical functions
     */

    /**
     * @ingroup simd_batch_bool_logical
     *
     * Computes the logical and of batches \c lhs and \c rhs.
     * @tparam X the actual type of boolean batch.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the logical and.
     */
    template <class X>
    inline X operator&&(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>& rhs)
    {
        return lhs() & rhs();
    }

    /**
     * @ingroup simd_batch_bool_logical
     *
     * Computes the logical and of the batch \c lhs and the scalar \c rhs.
     * Equivalent to the logical and of two boolean batches, where all the
     * values of the second one are initialized to \c rhs.
     * @tparam X the actual type of boolean batch.
     * @param lhs batch involved in the operation.
     * @param rhs boolean involved in the operation.
     * @return the result of the logical and.
     */
    template <class X>
    inline X operator&&(const simd_batch_bool<X>& lhs, bool rhs)
    {
        return lhs() & X(rhs);
    }

    /**
     * @ingroup simd_batch_bool_logical
     *
     * Computes the logical and of the scalar \c lhs and the batch \c rhs.
     * Equivalent to the logical and of two boolean batches, where all the
     * values of the first one are initialized to \c lhs.
     * @tparam X the actual type of boolean batch.
     * @param lhs boolean involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the logical and.
     */
    template <class X>
    inline X operator&&(bool lhs, const simd_batch_bool<X>& rhs)
    {
        return X(lhs) & rhs();
    }

    /**
     * @ingroup simd_batch_bool_logical
     *
     * Computes the logical or of batches \c lhs and \c rhs.
     * @tparam X the actual type of boolean batch.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the logical or.
     */
    template <class X>
    inline X operator||(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>& rhs)
    {
        return lhs() | rhs();
    }

    /**
     * @ingroup simd_batch_bool_logical
     *
     * Computes the logical or of the batch \c lhs and the scalar \c rhs.
     * Equivalent to the logical or of two boolean batches, where all the
     * values of the second one are initialized to \c rhs.
     * @tparam X the actual type of boolean batch.
     * @param lhs batch involved in the operation.
     * @param rhs boolean involved in the operation.
     * @return the result of the logical or.
     */
    template <class X>
    inline X operator||(const simd_batch_bool<X>& lhs, bool rhs)
    {
        return lhs() | X(rhs);
    }

    /**
     * @ingroup simd_batch_bool_logical
     *
     * Computes the logical or of the scalar \c lhs and the batch \c rhs.
     * Equivalent to the logical or of two boolean batches, where all the
     * values of the first one are initialized to \c lhs.
     * @tparam X the actual type of boolean batch.
     * @param lhs boolean involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the logical or.
     */
    template <class X>
    inline X operator||(bool lhs, const simd_batch_bool<X>& rhs)
    {
        return X(lhs) | rhs();
    }

    /*
     * @ingroup simd_batch_bool_logical
     *
     * Computes the logical not of \c rhs.
     * @tparam X the actual type of boolean batch.
     * @param rhs batch involved in the operation.
     * @return the result og the logical not.
     */
    template <class X>
    inline X operator!(const simd_batch_bool<X>& rhs)
    {
        return rhs() == X(false);
    }

    /*****************************
     * simd_batch implementation *
     *****************************/

    /**
     * @name Arithmetic computed assignment
     */
    //@{
    /**
     * Adds the batch \c rhs to \c this.
     * @param rhs the batch to add.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch<X>::operator+=(const X& rhs)
    {
        (*this)() = (*this)() + rhs;
        return (*this)();
    }

    /**
     * Adds the scalar \c rhs to each value contained in \c this.
     * @param rhs the scalar to add.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch<X>::operator+=(const value_type& rhs)
    {
        (*this)() = (*this)() + X(rhs);
        return (*this)();
    }

    /**
     * Substracts the batch \c rhs to \c this.
     * @param rhs the batch to substract.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch<X>::operator-=(const X& rhs)
    {
        (*this)() = (*this)() - rhs;
        return (*this)();
    }

    /**
     * Substracts the scalar \c rhs to each value contained in \c this.
     * @param rhs the scalar to substract.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch<X>::operator-=(const value_type& rhs)
    {
        (*this)() = (*this)() - X(rhs);
        return (*this)();
    }

    /**
     * Multiplies \c this with the batch \c rhs.
     * @param rhs the batch involved in the multiplication.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch<X>::operator*=(const X& rhs)
    {
        (*this)() = (*this)() * rhs;
        return (*this)();
    }

    /**
     * Multiplies each scalar contained in \c this with the scalar \c rhs.
     * @param rhs the scalar involved in the multiplication.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch<X>::operator*=(const value_type& rhs)
    {
        (*this)() = (*this)() * X(rhs);
        return (*this)();
    }

    /**
     * Divides \c this by the batch \c rhs.
     * @param rhs the batch involved in the division.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch<X>::operator/=(const X& rhs)
    {
        (*this)() = (*this)() / rhs;
        return (*this)();
    }

    /**
     * Divides each scalar contained in \c this by the scalar \c rhs.
     * @param rhs the scalar involved in the division.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch<X>::operator/=(const value_type& rhs)
    {
        (*this)() = (*this)() / X(rhs);
        return (*this)();
    }
    //@}

    /**
     * @name Bitwise computed assignment
     */
    /**
     * Assigns the bitwise and of \c rhs and \c this.
     * @param rhs the batch involved in the operation.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch<X>::operator&=(const X& rhs)
    {
        (*this)() = (*this)() & rhs;
        return (*this)();
    }

    /**
     * Assigns the bitwise or of \c rhs and \c this.
     * @param rhs the batch involved in the operation.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch<X>::operator|=(const X& rhs)
    {
        (*this)() = (*this)() | rhs;
        return (*this)();
    }

    /**
     * Assigns the bitwise xor of \c rhs and \c this.
     * @param rhs the batch involved in the operation.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch<X>::operator^=(const X& rhs)
    {
        (*this)() = (*this)() ^ rhs;
        return (*this)();
    }
    //@}

    /**
     * @name Increment and decrement operators
     */
    //@{
    /**
     * Pre-increment operator.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch<X>::operator++()
    {
        (*this)() += value_type(1);
        return (*this)();
    }

    /**
     * Post-increment operator.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch<X>::operator++(int)
    {
        X tmp = (*this)();
        (*this)() += value_type(1);
        return tmp;
    }

    /**
     * Pre-decrement operator.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch<X>::operator--()
    {
        (*this)() -= value_type(1);
        return (*this)();
    }

    /**
     * Post-decrement operator.
     * @return a reference to \c this.
     */
    template <class X>
    inline X& simd_batch<X>::operator--(int)
    {
        X tmp = (*this)();
        (*this)() -= value_type(1);
        return tmp;
    }
    //@}

    /**
     * @name Static downcast functions
     */
    //@{
    /**
     * Returns a reference to the actual derived type of the simd_batch_bool.
     */
    template <class X>
    inline X& simd_batch<X>::operator()()
    {
        return *static_cast<X*>(this);
    }

    /**
     * Returns a constant reference to the actual derived type of the simd_batch_bool.
     */
    template <class X>
    inline const X& simd_batch<X>::operator()() const
    {
        return *static_cast<const X*>(this);
    }
    //@}

    /**
     * @defgroup simd_batch_arithmetic Arithmetic operators
     */

    /**
     * @ingroup simd_batch_arithmetic
     *
     * Computes the sum of the batch \c lhs and the scalar \c rhs. Equivalent to the
     * sum of two batches where all the values of the second one are initialized to
     * \c rhs.
     * @tparam X the actual type of batch.
     * @param lhs batch involved in the addition.
     * @param rhs scalar involved in the addition.
     * @return the result of the addition.
     */
    template <class X>
    inline X operator+(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs)
    {
        return lhs() + X(rhs);
    }

    /**
     * @ingroup simd_batch_arithmetic
     *
     * Computes the sum of the scalar \c lhs and the batch \c rhs. Equivalent to the
     * sum of two batches where all the values of the first one are initialized to
     * \c rhs.
     * @tparam X the actual type of batch.
     * @param lhs scalar involved in the addition.
     * @param rhs batch involved in the addition.
     * @return the result of the addition.
     */
    template <class X>
    inline X operator+(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs)
    {
        return X(lhs) + rhs();
    }

    /**
     * @ingroup simd_batch_arithmetic
     *
     * Computes the difference of the batch \c lhs and the scalar \c rhs. Equivalent to the
     * difference of two batches where all the values of the second one are initialized to
     * \c rhs.
     * @tparam X the actual type of batch.
     * @param lhs batch involved in the difference.
     * @param rhs scalar involved in the difference.
     * @return the result of the difference.
     */
    template <class X>
    inline X operator-(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs)
    {
        return lhs() - X(rhs);
    }

    /**
     * @ingroup simd_batch_arithmetic
     *
     * Computes the difference of the scalar \c lhs and the batch \c rhs. Equivalent to the
     * difference of two batches where all the values of the first one are initialized to
     * \c rhs.
     * @tparam X the actual type of batch.
     * @param lhs scalar involved in the difference.
     * @param rhs batch involved in the difference.
     * @return the result of the difference.
     */
    template <class X>
    inline X operator-(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs)
    {
        return X(lhs) - rhs();
    }

    /**
     * @ingroup simd_batch_arithmetic
     *
     * Computes the product of the batch \c lhs and the scalar \c rhs. Equivalent to the
     * product of two batches where all the values of the second one are initialized to
     * \c rhs.
     * @tparam X the actual type of batch.
     * @param lhs batch involved in the product.
     * @param rhs scalar involved in the product.
     * @return the result of the product.
     */
    template <class X>
    inline X operator*(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs)
    {
        return lhs() * X(rhs);
    }

    /**
     * @ingroup simd_batch_arithmetic
     *
     * Computes the product of the scalar \c lhs and the batch \c rhs. Equivalent to the
     * difference of two batches where all the values of the first one are initialized to
     * \c rhs.
     * @tparam X the actual type of batch.
     * @param lhs scalar involved in the product.
     * @param rhs batch involved in the product.
     * @return the result of the product.
     */
    template <class X>
    inline X operator*(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs)
    {
        return X(lhs) * rhs();
    }

    /**
     * @ingroup simd_batch_arithmetic
     *
     * Computes the division of the batch \c lhs by the scalar \c rhs. Equivalent to the
     * division of two batches where all the values of the second one are initialized to
     * \c rhs.
     * @tparam X the actual type of batch.
     * @param lhs batch involved in the division.
     * @param rhs scalar involved in the division.
     * @return the result of the division.
     */
    template <class X>
    inline X operator/(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs)
    {
        return lhs() / X(rhs);
    }

    /**
     * @ingroup simd_batch_arithmetic
     *
     * Computes the division of the scalar \c lhs and the batch \c rhs. Equivalent to the
     * difference of two batches where all the values of the first one are initialized to
     * \c rhs.
     * @tparam X the actual type of batch.
     * @param lhs scalar involved in the division.
     * @param rhs batch involved in the division.
     * @return the result of the division.
     */
    template <class X>
    inline X operator/(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs)
    {
        return X(lhs) / rhs();
    }

    /**
     * @defgroup simd_batch_comparison Comparison operators
     */

    /**
     * @ingroup simd_batch_comparison
     *
     * Element-wise greater than comparison of batches \c lhs and \c rhs.
     * @tparam X the actual type of batch.
     * @param lhs batch involved in the comparison.
     * @param rhs batch involved in the comparison.
     * @return a boolean batch.
     */
    template <class X>
    inline typename simd_batch_traits<X>::batch_bool_type
    operator>(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        return rhs() < lhs();
    }

    /**
     * @ingroup simd_batch_comparison
     *
     * Element-wise greater or equal comparison of batches \c lhs and \c rhs.
     * @tparam X the actual type of batch.
     * @param lhs batch involved in the comparison.
     * @param rhs batch involved in the comparison.
     * @return a boolean batch.
     */
    template <class X>
    inline typename simd_batch_traits<X>::batch_bool_type
    operator>=(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        return rhs() <= lhs();
    }

    /**
     * Element-wise not of \c rhs.
     * @tparam X the actual type of batch.
     * @param rhs batch involved in the logical not operation.
     * @return boolean batch.
     */
    template <class X>
    inline typename simd_batch_traits<X>::batch_bool_type
    operator!(const simd_batch<X>& rhs)
    {
        return rhs() == X(0);
    }

    /**
     * Insert the batch \c rhs into the stream \c out.
     * @tparam X the actual type of batch.
     * @param out the output stream.
     * @param rhs the batch to output.
     * @return the output stream.
     */
    template <class X>
    inline std::ostream& operator<<(std::ostream& out, const simd_batch<X>& rhs)
    {
        out << '(';
        std::size_t s = simd_batch<X>::size;
        for (std::size_t i = 0; i < s - 1; ++i)
        {
            out << rhs()[i] << ", ";
        }
        out << rhs()[s - 1] << ')';
        return out;
    }

    /******************************************
     * generic batch operators implementation *
     ******************************************/

#define GENERIC_OPERATOR_IMPLEMENTATION(OP)        \
    using traits = simd_batch_traits<batch<T, N>>; \
    constexpr std::size_t align = traits::align;   \
    alignas(align) T tmp_lhs[N];                   \
    alignas(align) T tmp_rhs[N];                   \
    alignas(align) T tmp_res[N];                   \
    lhs.store_aligned(tmp_lhs);                    \
    rhs.store_aligned(tmp_rhs);                    \
    for (std::size_t i = 0; i < traits::size; ++i) \
    {                                              \
        tmp_res[i] = tmp_lhs[i] OP tmp_rhs[i];     \
    }                                              \
    return batch<T, N>(tmp_res, aligned_mode())

    template <class T, std::size_t N>
    inline batch<T, N> operator&&(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        GENERIC_OPERATOR_IMPLEMENTATION(&&);
    }

    template <class T, std::size_t N>
    inline batch<T, N> operator||(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        GENERIC_OPERATOR_IMPLEMENTATION(||);
    }

    template <class T, std::size_t N>
    inline batch<T, N> operator<<(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        GENERIC_OPERATOR_IMPLEMENTATION(<<);
    }

    template <class T, std::size_t N>
    inline batch<T, N> operator>>(const batch<T, N>& lhs, const batch<T, N>& rhs)
    {
        GENERIC_OPERATOR_IMPLEMENTATION(>>);
    }
}

#endif

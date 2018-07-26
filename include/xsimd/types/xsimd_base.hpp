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
#include <type_traits>

#include "../memory/xsimd_alignment.hpp"
#include "xsimd_utils.hpp"

namespace xsimd
{

    template <class T, size_t N>
    class batch;

    template <class T, std::size_t N>
    class batch_bool;

    namespace detail
    {
        template <class T, std::size_t N>
        struct batch_bool_kernel;

        template <class T, std::size_t N>
        struct batch_kernel;
    }

    template <class X>
    struct simd_batch_traits;

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

        using value_type = typename simd_batch_traits<X>::value_type;
        static constexpr std::size_t size = simd_batch_traits<X>::size;

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
    X operator&(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>&rhs);

    template <class X>
    X operator|(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>&rhs);

    template <class X>
    X operator^(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>&rhs);

    template <class X>
    X operator~(const simd_batch_bool<X>& rhs);

    template <class X>
    X bitwise_andnot(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>&rhs);

    template <class X>
    X operator==(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>&rhs);

    template <class X>
    X operator!=(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>&rhs);

    template <class X>
    bool all(const simd_batch_bool<X>& rhs);

    template <class X>
    bool any(const simd_batch_bool<X>& rhs);

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

    template <class X>
    std::ostream& operator<<(std::ostream& out, const simd_batch_bool<X>& rhs);

    /**************
     * simd_batch *
     **************/

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
        static constexpr std::size_t size = simd_batch_traits<X>::size;

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

        X& load_aligned(const char* src);
        X& load_unaligned(const char* src);

        void store_aligned(char* dst) const;
        void store_unaligned(char* dst) const;

    protected:

        simd_batch() = default;
        ~simd_batch() = default;

        simd_batch(const simd_batch&) = default;
        simd_batch& operator=(const simd_batch&) = default;

        simd_batch(simd_batch&&) = default;
        simd_batch& operator=(simd_batch&&) = default;
        
        using char_itype = typename std::conditional<std::is_signed<char>::value, int8_t, uint8_t>::type;
    };

    template <class X>
    X operator-(const simd_batch<X>& rhs);

    template <class X>
    X operator+(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator+(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs);

    template <class X>
    X operator+(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator-(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator-(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs);

    template <class X>
    X operator-(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator*(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator*(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs);

    template <class X>
    X operator*(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator/(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator/(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs);

    template <class X>
    X operator/(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs);

    template <class X>
    typename simd_batch_traits<X>::batch_bool_type
    operator==(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    typename simd_batch_traits<X>::batch_bool_type
    operator!=(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    typename simd_batch_traits<X>::batch_bool_type
    operator<(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    typename simd_batch_traits<X>::batch_bool_type
    operator<=(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    typename simd_batch_traits<X>::batch_bool_type
    operator>(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    typename simd_batch_traits<X>::batch_bool_type
    operator>=(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator&(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator|(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator^(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    X operator~(const simd_batch<X>& rhs);

    template <class X>
    X bitwise_andnot(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    typename simd_batch_traits<X>::batch_bool_type
    operator!(const simd_batch<X>& rhs);

    template <class X>
    X min(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    X max(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    X fmin(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    X fmax(const simd_batch<X>& lhs, const simd_batch<X>& rhs);

    template <class X>
    X abs(const simd_batch<X>& rhs);

    template <class X>
    X fabs(const simd_batch<X>& rhs);

    template <class X>
    X sqrt(const simd_batch<X>& rhs);

    template <class X>
    X fma(const simd_batch<X>& x, const simd_batch<X>& y, const simd_batch<X>& z);

    template <class X>
    X fms(const simd_batch<X>& x, const simd_batch<X>& y, const simd_batch<X>& z);

    template <class X>
    X fnma(const simd_batch<X>& x, const simd_batch<X>& y, const simd_batch<X>& z);

    template <class X>
    X fnms(const simd_batch<X>& x, const simd_batch<X>& y, const simd_batch<X>& z);

    template <class X>
    typename simd_batch_traits<X>::value_type 
    hadd(const simd_batch<X>& rhs);

    template <class X>
    X haddp(const simd_batch<X>* row);

    template <class X>
    X select(const typename simd_batch_traits<X>::batch_bool_type& cond, const simd_batch<X>& a, const simd_batch<X>& b);

    template <class X>
    typename simd_batch_traits<X>::batch_bool_type
    isnan(const simd_batch<X>& x);

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

    /**************************
     * bitwise cast functions *
     **************************/

    // Provides a reinterpret_case from batch<T_in, N_in> to batch<T_out, N_out>
    template <class B_in, class B_out>
    struct bitwise_cast_impl;

    // Shorthand for defining an intrinsic-based bitwise_cast implementation
    #define XSIMD_BITWISE_CAST_INTRINSIC(T_IN, N_IN, T_OUT, N_OUT, INTRINSIC)  \
        template <>                                                            \
        struct bitwise_cast_impl<batch<T_IN, N_IN>, batch<T_OUT, N_OUT>>       \
        {                                                                      \
            static inline batch<T_OUT, N_OUT> run(const batch<T_IN, N_IN>& x)  \
            {                                                                  \
                return INTRINSIC(x);                                           \
            }                                                                  \
        };


    // Backwards-compatible interface to bitwise_cast_impl
    template <class B, std::size_t N = simd_batch_traits<B>::size>
    B bitwise_cast(const batch<float, N>& x);

    template <class B, std::size_t N = simd_batch_traits<B>::size>
    B bitwise_cast(const batch<double, N>& x);

    template <class B, std::size_t N = simd_batch_traits<B>::size>
    B bitwise_cast(const batch<int32_t, N>& x);

    template <class B, std::size_t N = simd_batch_traits<B>::size>
    B bitwise_cast(const batch<int64_t, N>& x);

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
    * @defgroup simd_batch_bool_bitwise Bitwise functions
    */

    /**
     * @ingroup simd_batch_bool_bitwise
     *
     * Computes the bitwise and of batches \c lhs and \c rhs.
     * @tparam X the actual type of boolean batch.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise and.
     */
    template <class X>
    inline X operator&(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>&rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_bool_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::bitwise_and(lhs(), rhs());
    }

    /**
     * @ingroup simd_batch_bool_bitwise
     *
     * Computes the bitwise or of batches \c lhs and \c rhs.
     * @tparam X the actual type of boolean batch.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise or.
     */
    template <class X>
    inline X operator|(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>&rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_bool_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::bitwise_or(lhs(), rhs());
    }

    /**
     * @ingroup simd_batch_bool_bitwise
     *
     * Computes the bitwise xor of batches \c lhs and \c rhs.
     * @tparam X the actual type of boolean batch.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise xor.
     */
    template <class X>
    inline X operator^(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>&rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_bool_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::bitwise_xor(lhs(), rhs());
    }

    /**
     * @ingroup simd_batch_bool_bitwise
     *
     * Computes the bitwise not of batch \c rhs.
     * @tparam X the actual type of boolean batch.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise not.
     */
    template <class X>
    inline X operator~(const simd_batch_bool<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_bool_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::bitwise_not(rhs());
    }

    /**
     * @ingroup simd_batch_bool_bitwise
     *
     * Computes the bitwise and not of batches \c lhs and \c rhs.
     * @tparam X the actual type of boolean batch.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise and not.
     */
    template <class X>
    inline X bitwise_andnot(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>&rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_bool_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::bitwise_andnot(lhs(), rhs());
    }

    /**
     * @defgroup simd_batch_bool_comparison Comparison operators
     */

    /**
     * @ingroup simd_batch_bool_comparison
     *
     * Element-wise equality of batches \c lhs and \c rhs.
     * @param lhs batch involved in the comparison.
     * @param rhs batch involved in the comparison.
     * @return the result of the equality comparison.
     */
    template <class X>
    inline X operator==(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>&rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_bool_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::equal(lhs(), rhs());
    }

    /**
     * @ingroup simd_batch_bool_comparison
     *
     * Element-wise inequality of batches \c lhs and \c rhs.
     * @param lhs batch involved in the comparison.
     * @param rhs batch involved in the comparison.
     * @return the result of the inequality comparison.
     */
    template <class X>
    inline X operator!=(const simd_batch_bool<X>& lhs, const simd_batch_bool<X>&rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_bool_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::not_equal(lhs(), rhs());
    }

    /**
     * @defgroup simd_batch_bool_reducers Reducers
     */

    /**
     * @ingroup simd_batch_bool_reducers
     *
     * Returns true if all the boolean values in the batch are true,
     * false otherwise.
     * @param rhs the batch to reduce.
     * @return a boolean scalar.
     */
    template <class X>
    inline bool all(const simd_batch_bool<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_bool_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::all(rhs());
    }

    /**
     * @ingroup simd_batch_bool_reducers
     *
     * Return true if any of the boolean values in the batch is true,
     * false otherwise.
     * @param rhs the batch to reduce.
     * @return a boolean scalar.
     */
    template <class X>
    inline bool any(const simd_batch_bool<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_bool_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::any(rhs());
    }

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

    /**
     * Insert the batch \c rhs into the stream \c out.
     * @tparam X the actual type of batch.
     * @param out the output stream.
     * @param rhs the batch to output.
     * @return the output stream.
     */
    template <class X>
    inline std::ostream& operator<<(std::ostream& out, const simd_batch_bool<X>& rhs)
    {
        out << '(';
        std::size_t s = simd_batch_bool<X>::size;
        for (std::size_t i = 0; i < s - 1; ++i)
        {
            out << (rhs()[i] ? 'T' : 'F') << ", ";
        }
        out << (rhs()[s - 1] ? 'T' : 'F') << ')';
        return out;
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

    template <class X>
    inline X& simd_batch<X>::load_aligned(const char* src)
    {
        return (*this)().load_aligned(reinterpret_cast<const char_itype*>(src));
    }

    template <class X>
    inline X& simd_batch<X>::load_unaligned(const char* src)
    {
        return (*this)().load_unaligned(reinterpret_cast<const char_itype*>(src));
    }

    template <class X>
    void simd_batch<X>::store_aligned(char* dst) const
    {
        return (*this)().store_aligned(reinterpret_cast<char_itype*>(dst));
    }

    template <class X>
    void simd_batch<X>::store_unaligned(char* dst) const
    {
        return (*this)().store_unaligned(reinterpret_cast<char_itype*>(dst));
    }

    /**
     * @defgroup simd_batch_arithmetic Arithmetic operators
     */

    /**
     * @ingroup simd_batch_arithmetic
     *
     * Computes the opposite of the batch \c rhs.
     * @tparam X the actual type of batch.
     * @param rhs batch involved in the operation.
     * @return the opposite of \c rhs.
     */
    template <class X>
    inline X operator-(const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::neg(rhs());
    }
    
    /**
     * @ingroup simd_batch_arithmetic
     *
     * Computes the sum of the batches \c lhs and \c rhs.
     * @tparam X the actual type of batch.
     * @param lhs batch involved in the addition.
     * @param rhs batch involved in the addition.
     * @return the result of the addition.
     */
    template <class X>
    inline X operator+(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::add(lhs(), rhs());
    }

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
     * Computes the difference of the batches \c lhs and \c rhs.
     * @tparam X the actual type of batch.
     * @param lhs batch involved in the difference.
     * @param rhs batch involved in the difference.
     * @return the result of the difference.
     */
    template <class X>
    inline X operator-(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::sub(lhs(), rhs());
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
     * Computes the product of the batches \c lhs and \c rhs.
     * @tparam X the actual type of batch.
     * @param lhs batch involved in the product.
     * @param rhs batch involved in the product.
     * @return the result of the product.
     */
    template <class X>
    inline X operator*(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::mul(lhs(), rhs());
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
     * Computes the division of the batch \c lhs by the batch \c rhs.
     * @tparam X the actual type of batch.
     * @param lhs batch involved in the division.
     * @param rhs batch involved in the division.
     * @return the result of the division.
     */
    template <class X>
    inline X operator/(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::div(lhs(), rhs());
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
     * @ingroup simd_batch_arithmetic
     *
     * Computes the integer modulo of the batch \c lhs by the batch \c rhs.
     * @param lhs batch involved in the modulo.
     * @param rhs batch involved in the modulo.
     * @return the result of the modulo.
     */
    template <class X>
    inline X operator%(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::mod(lhs(), rhs());
    }

    /**
     * @ingroup simd_batch_arithmetic
     *
     * Computes the integer modulo of the batch \c lhs by the scalar \c rhs. Equivalent to the
     * modulo of two batches where all the values of the second one are initialized to
     * \c rhs.
     * @tparam X the actual type of batch.
     * @param lhs batch involved in the modulo.
     * @param rhs scalar involved in the modulo.
     * @return the result of the modulo.
     */
    template <class X>
    inline X operator%(const simd_batch<X>& lhs, const typename simd_batch_traits<X>::value_type& rhs)
    {
        return lhs() % X(rhs);
    }

    /**
     * @ingroup simd_batch_arithmetic
     *
     * Computes the integer modulo of the scalar \c lhs and the batch \c rhs. Equivalent to the
     * difference of two batches where all the values of the first one are initialized to
     * \c rhs.
     * @tparam X the actual type of batch.
     * @param lhs scalar involved in the modulo.
     * @param rhs batch involved in the modulo.
     * @return the result of the modulo.
     */
    template <class X>
    inline X operator%(const typename simd_batch<X>::value_type& lhs, const simd_batch<X>& rhs)
    {
        return X(lhs) % rhs();
    }

    /**
     * @defgroup simd_batch_comparison Comparison operators
     */

     /**
      * @ingroup simd_batch_comparison
      *
      * Element-wise equality comparison of batches \c lhs and \c rhs.
      * @param lhs batch involved in the comparison.
      * @param rhs batch involved in the comparison.
      * @return a boolean batch.
      */
    template <class X>
    inline typename simd_batch_traits<X>::batch_bool_type
    operator==(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::eq(lhs(), rhs());
    }

    /**
     * @ingroup simd_batch_comparison
     *
     * Element-wise inequality comparison of batches \c lhs and \c rhs.
     * @param lhs batch involved in the comparison.
     * @param rhs batch involved in the comparison.
     * @return a boolean batch.
     */
    template <class X>
    inline typename simd_batch_traits<X>::batch_bool_type
    operator!=(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::neq(lhs(), rhs());
    }

    /**
     * @ingroup simd_batch_comparison
     *
     * Element-wise lesser than comparison of batches \c lhs and \c rhs.
     * @param lhs batch involved in the comparison.
     * @param rhs batch involved in the comparison.
     * @return a boolean batch.
     */
    template <class X>
    inline typename simd_batch_traits<X>::batch_bool_type
    operator<(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::lt(lhs(), rhs());
    }

    /**
     * @ingroup simd_batch_comparison
     *
     * Element-wise lesser or equal to comparison of batches \c lhs and \c rhs.
     * @param lhs batch involved in the comparison.
     * @param rhs batch involved in the comparison.
     * @return a boolean batch.
     */
    template <class X>
    inline typename simd_batch_traits<X>::batch_bool_type
    operator<=(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::lte(lhs(), rhs());
    }

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
     * @defgroup simd_batch_bitwise Bitwise operators
     */

    /**
     * @ingroup simd_batch_bitwise
     *
     * Computes the bitwise and of the batches \c lhs and \c rhs.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise and.
     */
    template <class X>
    inline X operator&(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::bitwise_and(lhs(), rhs());
    }

    /**
     * @ingroup simd_batch_bitwise
     *
     * Computes the bitwise or of the batches \c lhs and \c rhs.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise or.
     */
    template <class X>
    inline X operator|(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::bitwise_or(lhs(), rhs());
    }
    
    /**
     * @ingroup simd_batch_bitwise
     *
     * Computes the bitwise xor of the batches \c lhs and \c rhs.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise xor.
     */
    template <class X>
    inline X operator^(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::bitwise_xor(lhs(), rhs());
    }

    /**
     * @ingroup simd_batch_bitwise
     *
     * Computes the bitwise not of the batches \c lhs and \c rhs.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise not.
     */
    template <class X>
    inline X operator~(const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::bitwise_not(rhs());
    }

    /**
     * @ingroup simd_batch_bitwise
     *
     * Computes the bitwise andnot of the batches \c lhs and \c rhs.
     * @param lhs batch involved in the operation.
     * @param rhs batch involved in the operation.
     * @return the result of the bitwise andnot.
     */
    template <class X>
    inline X bitwise_andnot(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::bitwise_andnot(lhs(), rhs());
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
     * Returns the smaller values of the batches \c lhs and \c rhs.
     * @param lhs a batch of integer or floating point values.
     * @param rhs a batch of integer or floating point values.
     * @return a batch of the smaller values.
     */
    template <class X>
    inline X min(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::min(lhs(), rhs());
    }

    /**
     * Returns the larger values of the batches \c lhs and \c rhs.
     * @param lhs a batch of integer or floating point values.
     * @param rhs a batch of integer or floating point values.
     * @return a batch of the larger values.
     */
    template <class X>
    inline X max(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::max(lhs(), rhs());
    }

    /**
     * Returns the smaller values of the batches \c lhs and \c rhs.
     * @param lhs a batch of floating point values.
     * @param rhs a batch of floating point values.
     * @return a batch of the smaller values.
     */
    template <class X>
    inline X fmin(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::fmin(lhs(), rhs());
    }

    /**
     * Returns the larger values of the batches \c lhs and \c rhs.
     * @param lhs a batch of floating point values.
     * @param rhs a batch of floating point values.
     * @return a batch of the larger values.
     */
    template <class X>
    inline X fmax(const simd_batch<X>& lhs, const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::fmax(lhs(), rhs());
    }
    
    /**
     * Computes the absolute values of each scalar in the batch \c rhs.
     * @param rhs batch of integer or floating point values.
     * @return the asbolute values of \c rhs.
     */
    template <class X>
    inline X abs(const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::abs(rhs());
    }

    /**
     * Computes the absolute values of each scalar in the batch \c rhs.
     * @param rhs batch floating point values.
     * @return the asbolute values of \c rhs.
     */
    template <class X>
    inline X fabs(const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::fabs(rhs());
    }

    /**
     * Computes the square root of the batch \c rhs.
     * @param rhs batch of floating point values.
     * @return the square root of \c rhs.
     */
    template <class X>
    inline X sqrt(const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::sqrt(rhs());
    }

    /**
     * Computes <tt>(x*y) + z</tt> in a single instruction when possible.
     * @param x a batch of integer or floating point values.
     * @param y a batch of integer or floating point values.
     * @param z a batch of integer or floating point values.
     * @return the result of the fused multiply-add operation.
     */
    template <class X>
    inline X fma(const simd_batch<X>& x, const simd_batch<X>& y, const simd_batch<X>& z)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::fma(x(), y(), z());
    }

    /**
     * Computes <tt>(x*y) - z</tt> in a single instruction when possible.
     * @param x a batch of integer or floating point values.
     * @param y a batch of integer or floating point values.
     * @param z a batch of integer or floating point values.
     * @return the result of the fused multiply-sub operation.
     */
    template <class X>
    inline X fms(const simd_batch<X>& x, const simd_batch<X>& y, const simd_batch<X>& z)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::fms(x(), y(), z());
    }

    /**
     * Computes <tt>-(x*y) + z</tt> in a single instruction when possible.
     * @param x a batch of integer or floating point values.
     * @param y a batch of integer or floating point values.
     * @param z a batch of integer or floating point values.
     * @return the result of the fused negated multiply-add operation.
     */
    template <class X>
    inline X fnma(const simd_batch<X>& x, const simd_batch<X>& y, const simd_batch<X>& z)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::fnma(x(), y(), z());
    }

    /**
     * Computes <tt>-(x*y) - z</tt> in a single instruction when possible.
     * @param x a batch of integer or floating point values.
     * @param y a batch of integer or floating point values.
     * @param z a batch of integer or floating point values.
     * @return the result of the fused negated multiply-sub operation.
     */
    template <class X>
    inline X fnms(const simd_batch<X>& x, const simd_batch<X>& y, const simd_batch<X>& z)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::fnms(x(), y(), z());
    }

    /**
     * @defgroup simd_batch_reducers Reducers
     */

    /**
     * @ingroup simd_batch_reducers
     *
     * Adds all the scalars of the batch \c rhs.
     * @param rhs batch involved in the reduction
     * @return the result of the reduction.
     */
    template <class X>
    inline typename simd_batch_traits<X>::value_type
    hadd(const simd_batch<X>& rhs)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::hadd(rhs());
    }

    /**
     * @ingroup simd_batch_reducers
     *
     * Parallel horizontal addition: adds the scalars of each batch
     * in the array pointed by \c row and store them in a returned
     * batch.
     * @param row an array of \c N batches
     * @return the result of the reduction.
     */
    template <class X>
    inline X haddp(const simd_batch<X>* row)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::haddp(row);
    }

    /**
     * @defgroup simd_batch_miscellaneous Miscellaneous
     */

    /**
     * @ingroup simd_batch_miscellaneous
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
    template <class X>
    inline X select(const typename simd_batch_traits<X>::batch_bool_type& cond, const simd_batch<X>& a, const simd_batch<X>& b)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::select(cond(), a(), b());
    }

    /**
     * Determines if the scalars in the given batch \c x are NaN values.
     * @param x batch of floating point values.
     * @return a batch of booleans.
     */
    template <class X>
    inline typename simd_batch_traits<X>::batch_bool_type
    isnan(const simd_batch<X>& x)
    {
        using value_type = typename simd_batch_traits<X>::value_type;
        using kernel = detail::batch_kernel<value_type, simd_batch_traits<X>::size>;
        return kernel::isnan(x());
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

    /*****************************************
     * bitwise cast functions implementation *
     *****************************************/

    template <class B, std::size_t N>
    B bitwise_cast(const batch<float, N>& x)
    {
        return bitwise_cast_impl<batch<float, N>, B>::run(x);
    }

    template <class B, std::size_t N>
    B bitwise_cast(const batch<double, N>& x)
    {
        return bitwise_cast_impl<batch<double, N>, B>::run(x);
    }

    template <class B, std::size_t N>
    B bitwise_cast(const batch<int32_t, N>& x)
    {
        return bitwise_cast_impl<batch<int32_t, N>, B>::run(x);
    }

    template <class B, std::size_t N>
    B bitwise_cast(const batch<int64_t, N>& x)
    {
        return bitwise_cast_impl<batch<int64_t, N>, B>::run(x);
    }
}

#endif

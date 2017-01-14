/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTEST_UTILS_HPP
#define XTEST_UTILS_HPP

#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <ostream>
#include <iomanip>
#include <string>
#include <typeinfo>

namespace xsimd
{

    template <class T>
    inline std::string value_type_name()
    {
        return typeid(T).name();
    }

    class vector4f;
    class vector2d;
    class vector8f;
    class vector4d;

    template <>
    inline std::string value_type_name<vector4f>()
    {
        return "vector4f";
    }

    template <>
    inline std::string value_type_name<vector2d>()
    {
        return "vector2d";
    }

    template <>
    inline std::string value_type_name<vector8f>()
    {
        return "vector8f";
    }

    template <>
    inline std::string value_type_name<vector4d>()
    {
        return "vector4d";
    }

    namespace detail
    {

        template <class T>
        bool check_is_small(const T& value, const T& tolerance)
        {
            using std::abs;
            return abs(value) < abs(tolerance);
        }

        template <class T>
        T safe_division(const T& lhs, const T& rhs)
        {
            if(rhs < static_cast<T>(1) && lhs > rhs * (std::numeric_limits<T>::max)())
            {
                return (std::numeric_limits<T>::max)();
            }
            if(lhs == static_cast<T>(0) ||
               rhs > static_cast<T>(1) &&
               lhs < rhs * (std::numeric_limits<T>::min)())
            {
                return static_cast<T>(0);
            }
            return lhs / rhs;
        }

        template <class T>
        bool check_is_close(const T& lhs, const T& rhs, const T& relative_precision)
        {
            using std::abs;
            T diff = abs(lhs - rhs);
            T d1 = safe_division(diff, abs(rhs));
            T d2 = safe_division(diff, abs(lhs));

            return d1 <= relative_precision && d2 <= relative_precision;
        }
    }

    template <class T>
    bool scalar_comparison(const T& lhs, const T& rhs)
    {
        using std::max;
        using std::abs;

        T relative_precision = 2048 * std::numeric_limits<T>::epsilon();
        T absolute_zero_prox = 2048 * std::numeric_limits<T>::epsilon();

        if(max(abs(lhs), abs(rhs)) < T(1e-3))
        {
            return detail::check_is_small(lhs - rhs, absolute_zero_prox);
        }
        else
        {
            return detail::check_is_close(lhs, rhs, relative_precision);
        }
    }

    template <class T, class A>
    bool vector_comparison(const std::vector<T, A>& lhs, const std::vector<T, A>& rhs)
    {
        if(lhs.size() != rhs.size())
        {
            return false;
        }
        for(auto lhs_iter = lhs.begin(), rhs_iter = rhs.begin(); lhs_iter != lhs.end(); ++lhs_iter, ++rhs_iter)
        {
            if(!scalar_comparison(*lhs_iter, *rhs_iter))
            {
                return false;
            }
        }
        return true;
    }

    template <class T, class A>
    inline std::ostream& operator<<(std::ostream& os, const std::vector<T, A>& v)
    {
        os << '[';
        for(size_t i = 0; i < v.size() - 1; ++i)
        {
            os << v[i] << ',';
        }
        os << v.back() << ']' << std::endl;
        return os;
    }

    template <class T>
    inline bool check_almost_equal(const T& res, const T& ref, std::ostream& out)
    {
        out << std::setprecision(20);
        if(scalar_comparison(res, ref))
        {
            out << "OK" << std::endl;
            return true;
        }
        else
        {
            out << "BAD" << std::endl;
            out << "Expected : " << std::endl << ref << std::endl;
            out << "Got      : " << std::endl << res << std::endl;
            return false;
        }
    }

    template <class T, class A>
    inline bool check_almost_equal(const std::vector<T, A>& res, const std::vector<T, A>& ref, std::ostream& out)
    {
        out << std::setprecision(20);
        if(vector_comparison(res, ref))
        {
            out << "OK" << std::endl;
            return true;
        }
        else
        {
            out << "BAD" << std::endl;
            out << "Expected : " << std::endl << ref;
            out << "Got      : " << std::endl << res << std::endl;
            return false;
        }
    }

}

#endif


/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_TEST_UTILS_HPP
#define XSIMD_TEST_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ostream>
#include <string>
#include <typeinfo>
#include <vector>

namespace xsimd
{

    template <class T>
    inline std::string value_type_name()
    {
        return typeid(T).name();
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
            if (rhs < static_cast<T>(1) && lhs > rhs * (std::numeric_limits<T>::max)())
            {
                return (std::numeric_limits<T>::max)();
            }
            if (lhs == static_cast<T>(0) ||
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

        if (max(abs(lhs), abs(rhs)) < T(1e-3))
        {
            return detail::check_is_small(lhs - rhs, absolute_zero_prox);
        }
        else
        {
            return detail::check_is_close(lhs, rhs, relative_precision);
        }
    }

    template <class T, class A>
    int vector_comparison(const std::vector<T, A>& lhs, const std::vector<T, A>& rhs)
    {
        if (lhs.size() != rhs.size())
        {
            return -1;
        }
        int nb_diff = 0;
        for (auto lhs_iter = lhs.begin(), rhs_iter = rhs.begin(); lhs_iter != lhs.end(); ++lhs_iter, ++rhs_iter)
        {
            if (!scalar_comparison(*lhs_iter, *rhs_iter))
            {
                ++nb_diff;
            }
        }
        return nb_diff;
    }

    template <class T, class A>
    inline std::ostream& operator<<(std::ostream& os, const std::vector<T, A>& v)
    {
        os << '[';
        for (size_t i = 0; i < v.size() - 1; ++i)
        {
            os << v[i] << ',';
        }
        os << v.back() << ']' << std::endl;
        return os;
    }

    template <class T>
    inline bool check_almost_equal(const std::string& topic, const T& res, const T& ref, std::ostream& out)
    {
        out << topic;
        out << std::setprecision(20);
        if (scalar_comparison(res, ref))
        {
            out << "OK" << std::endl;
            return true;
        }
        else
        {
            out << "BAD" << std::endl;
            out << "Expected : " << std::endl
                << ref << std::endl;
            out << "Got      : " << std::endl
                << res << std::endl;
            return false;
        }
    }

#define PRINT_COUT

    template <class T, class A>
    inline bool check_almost_equal(const std::string& topic, const std::vector<T, A>& res, const std::vector<T, A>& ref, std::ostream& out)
    {
        out << topic;
        out << std::setprecision(20);
        int comp = vector_comparison(res, ref);
        if (comp == 0)
        {
            out << "OK" << std::endl;
            return true;
        }
        else if (comp == -1)
        {
            out << "BAD" << std::endl;
            out << "Expected size : " << ref.size() << std::endl;
            out << "Actual size   : " << res.size() << std::endl;
#ifdef PRINT_COUT
            std::cout << topic << "BAD" << std::endl;
            std::cout << "Expected size : " << ref.size() << std::endl;
            std::cout << "Actual size   : " << res.size() << std::endl;
#endif
            return false;
        }
        else
        {
            double pct = double(comp) / (double(res.size()));
            out << "BAD" << std::endl;
            out << "Expected : " << std::endl
                << ref << std::endl;
            ;
            out << "Got      : " << std::endl
                << res << std::endl;
            out << "Nb diff  : " << comp << '(' << pct << "%)" << std::endl;
#ifdef PRINT_COUT
            std::cout << topic << "BAD" << std::endl;
            std::cout << "Nb diff  : " << comp << '(' << pct << "%)" << std::endl;
#endif
            return false;
        }
    }
}

#endif

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
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ostream>
#include <string>
#include <typeinfo>
#include <vector>

#ifdef XSIMD_ENABLE_XTL_COMPLEX
#include "xtl/xcomplex.hpp"
#endif

#define DEBUG_FLOAT_ACCURACY 0

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
            T d1 = safe_division(diff, T(abs(rhs)));
            T d2 = safe_division(diff, T(abs(lhs)));

            return d1 <= relative_precision && d2 <= relative_precision;
        }
    }

    template <class T>
    struct scalar_comparison
    {
        static bool run(const T& lhs, const T& rhs)
        {
            using std::max;
            using std::abs;

            // direct compare integers -- but need tolerance for inexact double conversion
            if (std::is_integral<T>::value && lhs < 10e6 && rhs < 10e6)
            {
                return lhs == rhs;
            }

            T relative_precision = 2048 * std::numeric_limits<T>::epsilon();
            T absolute_zero_prox = 2048 * std::numeric_limits<T>::epsilon();

            if (max(abs(lhs), abs(rhs)) < T(1e-3))
            {
                using res_type = decltype(lhs - rhs);
                return detail::check_is_small(lhs - rhs, res_type(absolute_zero_prox));
            }
            else
            {
                return detail::check_is_close(lhs, rhs, relative_precision);
            }
        }
    };

    template <class T>
    struct scalar_comparison<std::complex<T>>
    {
        static bool run(const std::complex<T>& lhs, const std::complex<T>& rhs)
        {
            using real_comparison = scalar_comparison<T>;
            return real_comparison::run(lhs.real(), rhs.real()) &&
                real_comparison::run(lhs.imag(), rhs.imag());
        }
    };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
    template <class T, bool i3ec>
    struct scalar_comparison<xtl::xcomplex<T, T, i3ec>>
    {
        static bool run(const xtl::xcomplex<T, T, i3ec>& lhs, const xtl::xcomplex<T, T, i3ec>& rhs)
        {
            using real_comparison = scalar_comparison<T>;
            return real_comparison::run(lhs.real(), rhs.real()) &&
                real_comparison::run(lhs.imag(), rhs.imag());
        }
    };
#endif

    template <class T, class A>
    int vector_comparison(const std::vector<T, A>& lhs, const std::vector<T, A>& rhs)
    {
        if (lhs.size() != rhs.size())
        {
            return -1;
        }
        int nb_diff = 0;
        int ind = 0;
        std::vector<int> indx_list;
        for (auto lhs_iter = lhs.begin(), rhs_iter = rhs.begin(); lhs_iter != lhs.end(); ++lhs_iter, ++rhs_iter)
        {
            ++ind;
            if (!scalar_comparison<T>::run(*lhs_iter, *rhs_iter))
            {
                ++nb_diff;
                indx_list.push_back(ind);
                if (nb_diff < 5)
                {
                    std::cout << ind << ": lhs = " << +(*lhs_iter) << " - rhs = " << +(*rhs_iter) << std::endl;
                }
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
            os << +v[i] << ',';
        }
        os << +v.back() << ']' << std::endl;
        return os;
    }

    template <class T>
    inline bool check_almost_equal(const std::string& topic, const T& res, const T& ref, std::ostream& out)
    {
        out << topic;
        out << std::setprecision(20);
        if (scalar_comparison<T>::run(res, ref))
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
            double pct = double(comp) / (double(res.size())) * 100.;
            out << "BAD" << std::endl;
            out << "Expected : " << std::endl
                << ref << std::endl;
            ;
            out << "Got      : " << std::endl
                << res << std::endl;
            out << "Nb diff  : " << comp << " (" << pct << "%)" << std::endl;
#ifdef PRINT_COUT
            std::cout << topic << "BAD" << std::endl;
            std::cout << "Nb diff  : " << comp << " (" << pct << "%)" << std::endl;
#endif
            return false;
        }
    }
}

#endif

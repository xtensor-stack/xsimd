#ifndef XSIMD_SCALAR_HPP
#define XSIMD_SCALAR_HPP

#include <cmath>
#include <limits>

namespace xsimd
{
    template<typename T, class = typename std::enable_if<std::is_scalar<T>::value>::type>
    T sadd(const T& lhs, const T& rhs)
    {
        if (std::numeric_limits<T>::is_signed)
        {
            if ((lhs > 0) && (rhs > std::numeric_limits<T>::max() - lhs))
            {
                return std::numeric_limits<T>::max();
            }
            else if ((lhs < 0) && (rhs < std::numeric_limits<T>::lowest() - lhs))
            {
                return std::numeric_limits<T>::lowest();
            }
            else {
                return lhs + rhs;
            }
        }
        else
        {
            if (rhs > std::numeric_limits<T>::max() - lhs)
            {
                return std::numeric_limits<T>::max();
            }
            else
            {
                return lhs + rhs;
            }

        }
    }

    template<typename T, class = typename std::enable_if<std::is_scalar<T>::value>::type>
    T ssub(const T& lhs, const T& rhs)
    {
        if (std::numeric_limits<T>::is_signed)
        {
            return sadd(lhs, (T)-rhs);
        }
        else
        {
            if (lhs < rhs)
            {
                return std::numeric_limits<T>::lowest();
            }
            else
            {
                return lhs - rhs;
            }

        }
    }

}

#endif

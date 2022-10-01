/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "xsimd/xsimd.hpp"
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

namespace xsimd
{

    template class batch<char>;
    template class batch<unsigned char>;
    template class batch<signed char>;
    template class batch<unsigned short>;
    template class batch<signed short>;
    template class batch<unsigned int>;
    template class batch<signed int>;
    template class batch<unsigned long>;
    template class batch<signed long>;
    template class batch<float>;
#if !XSIMD_WITH_NEON || XSIMD_WITH_NEON64
    template class batch<double>;
#endif
}
#endif

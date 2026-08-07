/****************************************************************************
 * Copyright (c) xsimd contributors                                         *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_KERNEL_FWD_HPP
#define XSIMD_KERNEL_FWD_HPP

#include "../types/xsimd_batch_fwd.hpp"

namespace xsimd::kernel
{
    template <class T, class A>
    batch<T, A> add(batch<T, A> lhs, batch<T, A> rhs) noexcept;
}

#endif

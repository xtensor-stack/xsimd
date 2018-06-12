/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_BUILDER_HPP
#define XSIMD_BUILDER_HPP

namespace xsimd
{
	template <class T>
	struct arange
	{
		using simd_type = typename simd_traits<T>::type;
		constexpr static std::size_t batch_size = simd_traits<T>::size;

		arange(T start, T step)
			: m_start(start), m_step(step)
		{
			T val = m_start;
			T steps[batch_size];
			for (auto it = std::begin(steps); it != std::end(steps); ++it)
			{
				*it = val;
				val += m_step;
			}
			xsimd::load_aligned(steps, m_stepper);
			m_step_increment = simd_type(m_step * batch_size);
		}

		simd_type operator[](std::size_t I)
		{
			return m_stepper + simd_type(I);
		}

		void store_aligned(T* dst, std::size_t size) const
		{
			std::size_t simd_size = size / simd_traits<T>::size;
			std::size_t simd_rest = size % simd_traits<T>::size;

			simd_type stepper = m_stepper;

			for (std::size_t i = 0; i < simd_size; ++i)
			{
				xsimd::store_aligned(dst, stepper);
				stepper += m_step_increment;
				dst += batch_size;
			}

			T val = simd_size != 0 ? dst[-1] : m_start;  // get last value
			for (std::size_t i = 0; i < simd_rest; ++i)
			{
				val += m_step;
				*(dst++) = val;
			}
		}

	private:
		simd_type m_stepper, m_step_increment;
		T m_start, m_step;
	};
}

#endif
/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 * Copyright (c) Marco Barbone                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_UTILS_COMPRESS_EXPAND_LUT_HPP
#define XSIMD_UTILS_COMPRESS_EXPAND_LUT_HPP

#include <cstddef>
#include <cstdint>

namespace xsimd
{
    namespace kernel
    {
        namespace utils
        {
            /*
             * Compile-time shuffle tables for compress / expand, for the
             * architectures that lack a native instruction. Row ``m`` of a
             * table holds the permutation for the lane mask ``m`` of a batch
             * with ``N`` lanes.
             */

            // One source byte index per byte of a 16 bytes register, for
            // byte shuffles (pshufb, tbl). Destination bytes that receive no
            // element hold 0x80, which both instructions turn into zero.
            template <std::size_t N>
            struct byte_shuffle_lut
            {
                static constexpr std::size_t lane_bytes = 16 / N;
                static constexpr std::size_t rows = std::size_t(1) << N;

                alignas(16) uint8_t expand[rows][16] = {};
                alignas(16) uint8_t compress[rows][16] = {};

                constexpr byte_shuffle_lut() noexcept
                {
                    for (std::size_t m = 0; m < rows; ++m)
                    {
                        for (std::size_t b = 0; b < 16; ++b)
                        {
                            expand[m][b] = 0x80;
                            compress[m][b] = 0x80;
                        }
                        std::size_t packed = 0;
                        for (std::size_t lane = 0; lane < N; ++lane)
                        {
                            if (((m >> lane) & 1u) == 0)
                                continue;
                            for (std::size_t b = 0; b < lane_bytes; ++b)
                            {
                                expand[m][lane * lane_bytes + b] = static_cast<uint8_t>(packed * lane_bytes + b);
                                compress[m][packed * lane_bytes + b] = static_cast<uint8_t>(lane * lane_bytes + b);
                            }
                            ++packed;
                        }
                    }
                }
            };

            template <std::size_t N>
            inline constexpr byte_shuffle_lut<N> byte_shuffle_lut_v {};

            // One source index per 32 bits slot of a 32 bytes register, for
            // vpermd. The eight 4 bits indices of a row are packed in a
            // uint32_t, slot ``k`` in bits ``[4k, 4k + 4)``. vpermd only reads
            // the low 3 bits of an index, so bit 3 is free: it is set for the
            // slots that receive an element, and cleared for the slots that
            // must be zeroed.
            template <std::size_t N>
            struct dword_shuffle_lut
            {
                static_assert(N == 4 || N == 8, "8 lanes of 32 bits or 4 lanes of 64 bits");
                static constexpr std::size_t lane_dwords = 8 / N;
                static constexpr std::size_t rows = std::size_t(1) << N;
                static constexpr uint32_t used = 0x8;

                alignas(16) uint32_t expand[rows] = {};
                alignas(16) uint32_t compress[rows] = {};

                constexpr dword_shuffle_lut() noexcept
                {
                    for (std::size_t m = 0; m < rows; ++m)
                    {
                        uint32_t e = 0;
                        uint32_t c = 0;
                        std::size_t packed = 0;
                        for (std::size_t lane = 0; lane < N; ++lane)
                        {
                            if (((m >> lane) & 1u) == 0)
                                continue;
                            for (std::size_t d = 0; d < lane_dwords; ++d)
                            {
                                const std::size_t dst = lane * lane_dwords + d;
                                const std::size_t src = packed * lane_dwords + d;
                                e |= (used | static_cast<uint32_t>(src)) << (4 * dst);
                                c |= (used | static_cast<uint32_t>(dst)) << (4 * src);
                            }
                            ++packed;
                        }
                        expand[m] = e;
                        compress[m] = c;
                    }
                }
            };

            template <std::size_t N>
            inline constexpr dword_shuffle_lut<N> dword_shuffle_lut_v {};
        }
    }
}

#endif

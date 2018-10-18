.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay 

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. raw:: html

   <style>
   .rst-content table.docutils {
       width: 100%;
       table-layout: fixed;
   }

   table.docutils .line-block {
       margin-left: 0;
       margin-bottom: 0;
   }

   table.docutils code.literal {
       color: initial;
   }

   code.docutils {
       background: initial;
   }
   </style>

Available wrappers
==================

The :ref:`batch <xsimd-batch-ref>` and :ref:`batch_bool <xsimd-batch-bool-ref>` generic template classes are not implemented
by default, only full specializations of these templates are available depending on the instruction set macros defined
according to the instruction sets provided by the compiler.

Fallback implementation
-----------------------

You may optionally enable a fallback implementation, which translates batch and batch_bool variants that do not exist in
hardware into scalar loops. This is done by setting the XSIMD_ENABLE_FALLBACK preprocessor flag before including any xsimd
header.

This scalar fallback enables you to test the correctness of your computations without having matching hardware available, but
you should be aware that it is only intended for use in validation scenarios. It has generally speaking not been tuned for
performance, and its run-time characteristics may vary enormously from one compiler to another. Enabling it in
performance-conscious production code is therefore strongly discouraged.

x86 architecture
----------------

Depending on the value of XSIMD_X86_INSTR_SET, the following wrappers are available:

- XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION

+--------------------------------+-------------------------------------+
| batch                          | batch_bool                          |
+================================+=====================================+
| batch<int8_t, 16>              | batch_bool<int8_t, 16>              |
+--------------------------------+-------------------------------------+
| batch<uint8_t, 16>             | batch_bool<uint8_t, 16>             |
+--------------------------------+-------------------------------------+
| batch<int16_t, 9>              | batch_bool<int16_t, 8>              |
+--------------------------------+-------------------------------------+
| batch<uint16_t, 8>             | batch_bool<uint16_t, 8>             |
+--------------------------------+-------------------------------------+
| batch<int32_t, 4>              | batch_bool<int32_t, 4>              |
+--------------------------------+-------------------------------------+
| batch<uint32_t, 4>             | batch_bool<uint32_t, 4>             |
+--------------------------------+-------------------------------------+
| batch<int64_t, 2>              | batch_bool<int64_t, 2>              |
+--------------------------------+-------------------------------------+
| batch<uint64_t, 2>             | batch_bool<uint64_t, 2>             |
+--------------------------------+-------------------------------------+
| batch<float, 4>                | batch_bool<float, 4>                |
+--------------------------------+-------------------------------------+
| batch<double, 2>               | batch_bool<double, 2>               |
+--------------------------------+-------------------------------------+
| batch<std::complex<float>, 4>  | batch_bool<std::complex<float>, 4>  |
+--------------------------------+-------------------------------------+
| batch<std::complex<double>, 2> | batch_bool<std::complex<double>, 2> |
+--------------------------------+-------------------------------------+

- XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION

In addition to the wrappers defined above, the following wrappers are available:

+--------------------------------+-------------------------------------+
| batch                          | batch_bool                          |
+================================+=====================================+
| batch<int8_t, 32>              | batch_bool<int8_t, 32>              |
+--------------------------------+-------------------------------------+
| batch<uint8_t, 32>             | batch_bool<uint8_t, 32>             |
+--------------------------------+-------------------------------------+
| batch<int16_t, 16>             | batch_bool<int16_t, 16>             |
+--------------------------------+-------------------------------------+
| batch<uint16_t, 16>            | batch_bool<uint16_t, 16>            |
+--------------------------------+-------------------------------------+
| batch<int32_t, 8>              | batch_bool<int32_t, 8>              |
+--------------------------------+-------------------------------------+
| batch<uint32_t, 8>             | batch_bool<uint32_t, 8>             |
+--------------------------------+-------------------------------------+
| batch<int64_t, 4>              | batch_bool<int64_t, 4>              |
+--------------------------------+-------------------------------------+
| batch<uint64_t, 4>             | batch_bool<uint64_t, 4>             |
+--------------------------------+-------------------------------------+
| batch<float, 8>                | batch_bool<float, 8>                |
+--------------------------------+-------------------------------------+
| batch<double, 4>               | batch_bool<double, 4>               |
+--------------------------------+-------------------------------------+
| batch<std::complex<float>, 8>  | batch_bool<std::complex<float>, 8>  |
+--------------------------------+-------------------------------------+
| batch<std::complex<double>, 4> | batch_bool<std::complex<double>, 4> |
+--------------------------------+-------------------------------------+

- XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX512_VERSION

In addition to the wrappers defined above, the following wrappers are available:

+--------------------------------+-------------------------------------+
| batch                          | batch_bool                          |
+================================+=====================================+
| batch<int8_t, 64>              | batch_bool<int8_t, 64>              |
+--------------------------------+-------------------------------------+
| batch<uint8_t, 64>             | batch_bool<uint8_t, 64>             |
+--------------------------------+-------------------------------------+
| batch<int16_t, 32>             | batch_bool<int16_t, 32>             |
+--------------------------------+-------------------------------------+
| batch<uint16_t, 32>            | batch_bool<uint16_t, 32>            |
+--------------------------------+-------------------------------------+
| batch<int32_t, 16>             | batch_bool<int32_t, 16>             |
+--------------------------------+-------------------------------------+
| batch<uint32_t, 16>            | batch_bool<uint32_t, 16>            |
+--------------------------------+-------------------------------------+
| batch<int64_t, 8>              | batch_bool<int64_t, 8>              |
+--------------------------------+-------------------------------------+
| batch<uint64_t, 8>             | batch_bool<uint64_t, 8>             |
+--------------------------------+-------------------------------------+
| batch<float, 16>               | batch_bool<float, 16>               |
+--------------------------------+-------------------------------------+
| batch<double, 8>               | batch_bool<double, 8>               |
+--------------------------------+-------------------------------------+
| batch<std::complex<float>, 16> | batch_bool<std::complex<float>, 16> |
+--------------------------------+-------------------------------------+
| batch<std::complex<double>, 8> | batch_bool<std::complex<double>, 8> |
+--------------------------------+-------------------------------------+

ARM architecture
----------------

Depending on the value of XSIMD_ARM_INSTR_SET, the following wrappers are available:

- XSIMD_ARM_INSTR_SET >= XSIMD_ARM7_NEON_VERSION

+--------------------------------+-------------------------------------+
| batch                          | batch_bool                          |
+================================+=====================================+
| batch<int8_t, 16>              | batch_bool<int8_t, 16>              |
+--------------------------------+-------------------------------------+
| batch<uint8_t, 16>             | batch_bool<uint8_t, 16>             |
+--------------------------------+-------------------------------------+
| batch<int16_t, 8>              | batch_bool<int16_t, 8>              |
+--------------------------------+-------------------------------------+
| batch<uint16_t, 8>             | batch_bool<uint16_t, 8>             |
+--------------------------------+-------------------------------------+
| batch<int32_t, 4>              | batch_bool<int32_t, 4>              |
+--------------------------------+-------------------------------------+
| batch<uint32_t, 4>             | batch_bool<uint32_t, 4>             |
+--------------------------------+-------------------------------------+
| batch<int64_t, 2>              | batch_bool<int64_t, 2>              |
+--------------------------------+-------------------------------------+
| batch<uint64_t, 2>             | batch_bool<uint64_t, 2>             |
+--------------------------------+-------------------------------------+
| batch<float, 4>                | batch_bool<float, 4>                |
+--------------------------------+-------------------------------------+
| batch<std::complex<float>, 4>  | batch_bool<std::complex<float>, 4>  |
+--------------------------------+-------------------------------------+

- XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION

In addition to the wrappers defined above, the following wrappers are available:

+--------------------------------+-------------------------------------+
| batch                          | batch_bool                          |
+================================+=====================================+
| batch<double, 2>               | batch_bool<double, 2>               |
+--------------------------------+-------------------------------------+
| batch<std::complex<double>, 2> | batch_bool<std::complex<double>, 2> |
+--------------------------------+-------------------------------------+

.. warning::

   Support for ``std::complex`` on ARM is still experimental. You may
   experience accuracy errors with ``std::complex<float>``.

XTL complex support
-------------------

If the preprocessor token ``XSIMD_ENABLE_XTL_COMPLEX`` is defined, ``xsimd``
provides batches for ``xtl::xcomplex``, similar to those for ``std::complex``.
This requires ``xtl`` to be installed.


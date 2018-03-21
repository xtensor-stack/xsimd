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

The :ref:`batch <xsimd-batch-ref>` and :ref:`batch_bool <xsimd-batch-bool-ref>` generic template classes are not implemented,
only full specializations of these templates are available depending on the instruction set macros defined according to
the instruction sets provided by the compiler.

x86 architecture
----------------

Depending on the value of XSIMD_X86_INSTR_SET, the following wrappers are available:

- XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION

+-------------------+------------------------+
| batch             | batch_bool             |
+===================+========================+
| batch<float, 4>   | batch_bool<float, 4>   |
+-------------------+------------------------+
| batch<int32_t, 4> | batch_bool<int32_t, 4> |
+-------------------+------------------------+
| batch<double, 2>  | batch_bool<double, 2>  |
+-------------------+------------------------+
| batch<int64_t, 2> | batch_bool<int64_t, 2> |
+-------------------+------------------------+

- XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION

In addition to the wrappers defined above, the following wrappers are available:

+-------------------+------------------------+
| batch             | batch_bool             |
+===================+========================+
| batch<float, 8>   | batch_bool<float, 8>   |
+-------------------+------------------------+
| batch<int32_t, 8> | batch_bool<int32_t, 8> |
+-------------------+------------------------+
| batch<double, 4>  | batch_bool<double, 4>  |
+-------------------+------------------------+
| batch<int64_t, 4> | batch_bool<int64_t, 4> |
+-------------------+------------------------+

ARM architecture
----------------

Depending on the value of XSIMD_ARM_INSTR_SET, the following wrappers are available:

- XSIMD_ARM_INSTR_SET >= XSIMD_ARM7_NEON_VERSION

+-------------------+------------------------+
| batch             | batch_bool             |
+-------------------+------------------------+
| batch<float, 4>   | batch_bool<float, 4>   |
+-------------------+------------------------+
| batch<int32_t, 4> | batch_bool<int32_t, 4> |
+-------------------+------------------------+
| batch<int64_t, 2> | batch_bool<int64_t, 2> |
+-------------------+------------------------+

- XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION

In addition to the wrappers defined above, the following wrapper is available:

+-------------------+------------------------+
| batch             | batch_bool             |
+-------------------+------------------------+
| batch<double, 2>  | batch_bool<double, 2>  |
+-------------------+------------------------+

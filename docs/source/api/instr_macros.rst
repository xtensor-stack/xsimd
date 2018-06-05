.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

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

Instruction set macros
======================

`xsimd` defines different macros depending on the symbols defined by the compiler options.

x86 architecture
----------------

If one of the following symbols is detected, XSIMD_X86_INSTR_SET is set to the corresponding version and
XSIMD_X86_INSTR_SET_AVAILABLE is defined.

+-------------------+-----------------------------+
| Symbol            | Version                     |
+===================+=============================+
| __SSE__           | XSIMD_X86_SSE_VERSION       |
+-------------------+-----------------------------+
| _M_IX86_FP >= 1   | XSIMD_X86_SSE_VERSION       |
+-------------------+-----------------------------+
| __SSE2__          | XSIMD_X86_SSE2_VERSION      |
+-------------------+-----------------------------+
| _M_X64            | XSIMD_X86_SSE2_VERSION      |
+-------------------+-----------------------------+
| _M_IX86_FP >= 2   | XSIMD_X86_SSE2_VERSION      |
+-------------------+-----------------------------+
| __SSE3__          | XSIMD_X86_SSE3_VERSION      |
+-------------------+-----------------------------+
| __SSSE3__         | XSIMD_X86_SSSE3_VERSION     |
+-------------------+-----------------------------+
| __SSE4_1__        | XSIMD_X86_SSE4_1_VERSION    |
+-------------------+-----------------------------+
| __SSE4_2__        | XSIMD_X86_SSE4_2_VERSION    |
+-------------------+-----------------------------+
| __AVX__           | XSIMD_X86_AVX_VERSION       |
+-------------------+-----------------------------+
| __FMA__           | XSIMD_X86_FMA3_VERSION      |
+-------------------+-----------------------------+
| __AVX2__          | XSIMD_X86_AVX2_VERSION      |
+-------------------+-----------------------------+
| __AVX512__        | XSIMD_X86_AVX512_VERSION    |
+-------------------+-----------------------------+
| __KNCNI__         | XSIMD_X86_AVX512_VERSION    |
+-------------------+-----------------------------+
| __AVX512F__       | XSIMD_X86_AVX512_VERSION    |
+-------------------+-----------------------------+

x86_AMD architecture
--------------------

If one of the following symbols is detected, XSIMD_X86_AMD_INSTR_SET is set to the corresponding version and
XSIMD_X86_AMD_SET_AVAILABLE is defined.

+-------------------+-----------------------------+
| Symbol            | Version                     |
+===================+=============================+
| __SSE4A__         | XSIMD_X86_AMD_SSE4A_VERSION |
+-------------------+-----------------------------+
| __FMA__           | XSIMD_X86_AMD_FMA4_VERSION  |
+-------------------+-----------------------------+
| __XOP__           | XSIMD_X86_AMD_XOP_VERSION   |
+-------------------+-----------------------------+

If one of the previous symbol is defined, other x86 instruction sets not specific to AMD should be available too;
thus XSIMD_X86_INSTR_SET and XSIMD_X86_INSTR_SET_AVAILABLE should be defined. In that case, XSIMD_X86_AMD_INSTR_SET
is set to the maximum of XSIMD_X86_INSTR_SET and the current value of XSIMD_X86_AMD_INSTR_SET.

PPC architecture
----------------

If one of the following symbols is detected, XSIMD_PPC_INSTR_SET is set to the corresponding version and
XSIMD_PPC_INSTR_AVAILABLE is defined.

+-------------------+-----------------------------+
| Symbol            | Version                     |
+===================+=============================+
| __ALTIVEC__       | XSIMD_PPC_VMX_VERSION       |
+-------------------+-----------------------------+
| __VEC__           | XSIMD_PPC_VMX_VERSION       |
+-------------------+-----------------------------+
| __VSX__           | XSIMD_PPC_VSX_VERSION       |
+-------------------+-----------------------------+
| __VECTOR4DOUBLE__ | XSIMD_PPC_QPX_VERSION       |
+-------------------+-----------------------------+

ARM architecture
----------------

If one of the following condition is detected, XSIMD_ARM_INSTR_SET is set to the corresponding version and
XSIMD_ARM_INSTR_AVAILABLE is defined.

+-------------------+-----------------------------+
| Symbol            | Version                     |
+===================+=============================+
| __ARM_ARCH == 7   | XSIMD_ARM7_NEON_VERSION     |
+-------------------+-----------------------------+
| __ARM_ARCH == 8   | XSIMD_ARM8_32_NEON_VERSION  |
| && ! __aarch64__  |                             |
+-------------------+-----------------------------+
| __ARM_ARCH == 8   | XSIMD_ARM8_64_NEON_VERSION  |
| && __aarch64__    |                             |
+-------------------+-----------------------------+

Generic instruction set
-----------------------

If XSIMD_*_INSTR_SET_AVAILABLE has been defined as explained above, XSIMD_INSTR_SET is set to XSIMD_*_INSTR_SET
and XSIMD_INSTR_SET_AVAILABLE is defined.


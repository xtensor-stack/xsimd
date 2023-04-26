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

Each of these macros corresponds to an instruction set supported by XSIMD. They
can be used to filter arch-specific code.

.. doxygengroup:: xsimd_config_macro
   :project: xsimd
   :content-only:

Changing Default architecture
*****************************

You can change the default instruction set used by xsimd (when none is provided
explicitely) by setting the ``XSIMD_DEFAULT_ARCH`` macro to, say, ``xsimd::avx2``.
A common usage is to set it to ``xsimd::unsupported`` as a way to detect
instantiation of batches with the default architecture.

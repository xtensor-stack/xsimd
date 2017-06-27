.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. image:: xsimd.svg
   :alt: xsimd

C++ wrappers for SIMD intrinsics.

Introduction
------------

`xsimd` is a library meant for SIMD (Single Instruction, Multiple Data) programming. It provides C++ wrappers for SIMD intrinsics and an implementation
of common mathematical functions based on these wrappers. You can find out more about this implementation of C++ wrappers for SIMD intrinsics at the
`The C++ Scientist`_. The mathematical functions are a lightweight implementation of `boost.SIMD`_.

`xsimd` requires a C++14 compliant compiler. The following C++ compilers are supported:

+-------------------------+-------------------------------+
| Compiler                | Version                       |
+=========================+===============================+
| Microsoft Visual Studio | MSVC 2015 update 2 and above  |
+-------------------------+-------------------------------+
| g++                     | 4.9 and above                 |
+-------------------------+-------------------------------+
| clang                   | 3.7 and above                 |
+-------------------------+-------------------------------+

The following SIMD instruction set extensions are supported:

+--------------+----------------------------------------------------+
| Architecture | Instruction set extensions                         |
+==============+====================================================+
| x86          | SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, FMA3, AVX2 |
+--------------+----------------------------------------------------+
| x86 AMD      | same as above + SSE4A, FMA4, XOP                   |
+--------------+----------------------------------------------------+

Licensing
---------

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions.

This software is licensed under the BSD-3-Clause license. See the LICENSE file for details.


.. toctree::
   :caption: INSTALLATION
   :maxdepth: 2

   installation

.. toctree::
   :caption: USAGE
   :maxdepth: 2

   basic_usage

.. toctree::
   :caption: API REFERENCE
   :maxdepth: 2
   
   api/instr_macros
   api/batch_index
   api/math_index
   api/aligned_allocator

.. _The C++ Scientist: http://johanmabille.github.io/blog/archives/
.. _boost.SIMD: https://github.com/NumScale/boost.simd


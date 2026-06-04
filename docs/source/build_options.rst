.. Copyright (c) 2016, xsimd contributors

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.


Build Options
=============

Some macros can be set to configure the behaviour of xsimd at compulte time.

Changing Default Architecture
-----------------------------

You can change the default instruction set used by xsimd (when none is provided
explicitly) by setting the ``XSIMD_DEFAULT_ARCH`` macro to, say, ``xsimd::avx2``.
A common usage is to set it to ``xsimd::unsupported`` as a way to detect
instantiation of batches with the default architecture.

This will change the value of ``xsimd::default_arch`` used as default template parameter for
:ref:`batch <xsimd-batch-ref>`.
See :ref:`arch-manipulation` for all available architectures.

In CMake, this can be done with the following.

.. code:: cmake

    target_add_compile_definitions(myproject PRIVATE XSIMD_DEFAULT_ARCH=xsimd::avx2)


Enabling emulated architecture
------------------------------

When the compiler macro ``XSIMD_WITH_EMULATED`` is set to ``1``, xsimd also
exhibits an emulated architecture.

See :ref:`emulated-mode` for all available architectures.

In CMake, this can be done with the following.

.. code:: cmake

    target_add_compile_definitions(myproject PRIVATE XSIMD_WITH_EMULATED=1)


Enabling complex support with ``xtl``
-------------------------------------

If the preprocessor token ``XSIMD_ENABLE_XTL_COMPLEX`` is defined, ``xsimd``
provides constructors of ``xsimd::batch<std::complex<T>, A>`` from
``xtl::xcomplex``, similar to those for ``std::complex``. 
This requires the `xtl <https://github.com/xtensor-stack/xtl>`_ library to be installed
and in the project include directories.

In CMake, this can be done with an option before the package search (xsimd 14.3 onwards).
It will link the ``xtl`` library as well as adding the macro definition to ``xsimd`` interface.

.. code:: cmake

    option(XSIMD_ENABLE_XTL_COMPLEX "Enable xtl complex support" ON)
    find_package(xsimd REQUIRED)

.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay 

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Batch of Complex Numbers
========================

.. doxygenclass:: xsimd::batch< std::complex< T >, A >
   :project: xsimd
   :members:

Operations Specific to Batches of Complex Numbers
-------------------------------------------------

.. doxygengroup:: batch_complex
   :project: xsimd
   :content-only:

XTL Complex Support
-------------------

If the preprocessor token ``XSIMD_ENABLE_XTL_COMPLEX`` is defined, ``xsimd``
provides constructors of ``xsimd::batch< std::complex< T >, A >`` from
``xtl::xcomplex``, similar to those for ``std::complex``.  This requires `XTL`_
to be installed.

.. _XTL: https://github.com/xtensor-stack/xtl

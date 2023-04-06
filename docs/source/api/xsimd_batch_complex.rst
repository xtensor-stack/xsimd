.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay 

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Batch of complex numbers
========================

.. doxygenclass:: xsimd::batch< std::complex< T >, A >
   :project: xsimd
   :members:

Operations specific to batches of complex numbers
-------------------------------------------------

.. doxygengroup:: batch_complex_op
   :project: xsimd
   :content-only:

XTL complex support
-------------------

If the preprocessor token ``XSIMD_ENABLE_XTL_COMPLEX`` is defined, ``xsimd``
provides constructors of ``xsimd::batch< std::complex< T >, A >`` from
``xtl::xcomplex``, similar to those for ``std::complex``.  This requires ``xtl``
to be installed.

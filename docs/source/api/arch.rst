.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay 

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Architecture manipulation
=========================

xsimd provides an high level description of the instruction sets it manipulates.
The mentioned types are primarily used as template parameters for :ref:`batch
<xsimd-batch-ref>`, and when interacting with :cpp:func:`xsimd::dispatch()`.

The best available architecture is available at compile time through
``xsimd::best_arch`` which also happens to be ``xsimd::default_arch``.

.. doxygengroup:: architectures
   :project: xsimd
   :members:

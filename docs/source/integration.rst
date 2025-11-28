.. Copyright (c) 2025, Serge Guelton

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Integration
===========

When Targeting a Single Architecture
------------------------------------

If you compile your whole project for a single architecture, you can rely on the
implicit architecture parameter for :cpp:class:`xsimd::batch`. Just add your
source using `xsimd` to your project build system, pass down the
appropriate flags and the magic should happen.

It's very common though to have a base application with minimal architectural
constraints, while still wanting to benefit from the acceleration of better
instruction sets if those are available.

When Targeting Multiple Architectures
-------------------------------------

It's very common, especially when targeting Intel hardware, to set a minimal
baseline, say SSE2, for the base application, while still shipping computation
kernels specialized for SSE4.2, AVX2 or AVX512BF.

In that case one can write specific kernels for each targeted instruction set
(or a generic one that's instantiated for each targeted instruction set). Those
kernels must then be compiled with the appropriate flags independently, and
linked into the application.

`xsimd` provides a generic dispatch mechanism that can be used from the *base
application* to pick the best kernel *at runtime* based on runtime detection of the
supported architectures, as described more in detailed in :ref:`Arch
Dispatching`.

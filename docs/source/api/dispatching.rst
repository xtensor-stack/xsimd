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

.. _Arch Dispatching:

Arch Dispatching
================

`xsimd` provides a generic way to dispatch a function call based on the architecture the code was compiled for and the architectures available at runtime.
The :cpp:func::`xsimd::dispatch` function takes a functor whose call operator takes an architecture parameter as first operand, followed by any number of arguments ``Args...`` and turn it into a
dispatching functor that takes ``Args...`` as arguments.

.. doxygenfunction:: xsimd::dispatch
    :project: xsimd

Following code showcases a usage of the :cpp:func::`xsimd::dispatch` function:

.. code-block:: c++

    // functor with a call method that depends on `Arch`
    struct sum {
      template<class Arch, class T>
      T operator()(Arch, T const* data, unsigned size)
      {
        using batch = xsimd::batch<T, Arch>;
        batch acc(static_cast<T>(0));
        const unsigned n = size / batch::size * batch::size;
        for(unsigned i = 0; i != n; i += batch::size)
            acc += batch::load_unaligned(data + i);
        T star_acc = xsimd::hadd(acc);
        for(unsigned i = n; i < size; ++i)
          star_acc += data[i];
        return star_acc;
      }
    };

    // Create the dispatching function.
    auto dispatched = xsimd::dispatch(sum{});

    // Call the appropriate implementation based on runtime information.
    float res = dispatched(data, 17);

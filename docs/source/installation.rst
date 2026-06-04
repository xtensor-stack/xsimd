.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.


.. raw:: html

   <style>
   .rst-content .section>img,
   .rst-content section>img {
       width: 30px;
       margin-bottom: 0;
       margin-top: 0;
       margin-right: 15px;
       margin-left: 15px;
       float: left;
   }
   </style>

Installation
============

Getting the library
-------------------

`xsimd` is a header-only library, so installing it is just a matter of copying the ``include/xsimd`` directory.

However we provide standardized means to install it, with package managers or with cmake.
Besides the `xsimd` headers, all these methods place the ``cmake`` project configuration file in the right location so that third-party projects can use cmake's ``find_package`` to locate `xsimd` headers.

.. image:: conda.svg

Using the conda-forge Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A package for `xsimd` is available for the `mamba <https://mamba.readthedocs.io>`_ (or `conda <https://conda.io>`_) package manager.

.. code:: bash

    mamba install -c conda-forge xsimd

.. image:: spack.svg

Using the Spack Package
~~~~~~~~~~~~~~~~~~~~~~~

A package for `xsimd` is available on the `Spack <https://spack.io>`_ package manager.

.. code:: bash

    spack install xsimd
    spack load xsimd

.. image:: cmake.svg

From Source with cmake
~~~~~~~~~~~~~~~~~~~~~~

You can install `xsimd` from source with `Cmake <https://cmake.org/>`_. From the source directory:

.. code:: bash

    cmake -B build/ -D CMAKE_INSTALL_PREFIX=/path/to/install/dir
    cmake --build build/
    cmake --install build/

You may need to customize the default `CMake <https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html>`_, for instance for picking ``nmake`` on Windows platforms:

.. code:: bash

    cmake -B build/ -G "NMake Makefiles" -D CMAKE_INSTALL_PREFIX=/path/to/install/dir
    cmake --build build/
    cmake --install build/


Using the library inside CMake
------------------------------

Inside the user's ``CMakelists.txt``, the user can add xsimd to their build with the ``xsimd``
target.
When using CMake this way, this will set add the xsimd headers in the compiler options, as well
as forward other xsimd requirements (minimum C++ version for instance).

.. code:: cmake

    find_package(xsimd REQUIRED)
    target_link_libraries(myproject PRIVATE xsimd)

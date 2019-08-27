.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.


.. raw:: html

   <style>
   .rst-content .section>img {
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

Although ``xsimd`` is a header-only library, we provide standardized means to install it, with package managers or with cmake.

Besides the xsimd headers, all these methods place the ``cmake`` project configuration file in the right location so that third-party projects can use cmake's ``find_package`` to locate xsimd headers.

.. image:: conda.svg

Using the conda package
-----------------------

A package for xsimd is available on the conda package manager.

.. code::

    conda install -c conda-forge xsimd 

Using the Conan package
-----------------------

If you are using Conan to manage your dependencies, merely add `xsimd/x.y.z@omaralvarez/public-conan` to your requires, where x.y.z
is the release version you want to use. Please file issues in [conan-xsimd](https://github.com/omaralvarez/conan-xsimd) if you
experience problems with the packages. Sample `conanfile.txt`:

.. code::

    [requires]
    xsimd/7.2.3@omaralvarez/public-conan

    [generators]
    cmake

.. image:: spack.svg

Using the Spack package
-----------------------

A package for xsimd is available on the Spack package manager.

.. code::

    spack install xsimd
    spack load xsimd

.. image:: cmake.svg

From source with cmake
----------------------

You can also install ``xsimd`` from source with cmake. On Unix platforms, from the source directory:

.. code::

    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=/path/to/prefix ..
    make install

On Windows platforms, from the source directory:

.. code::

    mkdir build
    cd build
    cmake -G "NMake Makefiles" -DCMAKE_INSTALL_PREFIX=/path/to/prefix ..
    nmake
    nmake install

# ![xsimd](http://quantstack.net/assets/images/xsimd.svg)

[![Travis](https://travis-ci.org/QuantStack/xsimd.svg?branch=master)](https://travis-ci.org/QuantStack/xsimd)
[![Appveyor](https://ci.appveyor.com/api/projects/status/unk39xhu0wxmiif7?svg=true)](https://ci.appveyor.com/project/QuantStack/xsimd)
[![Documentation Status](http://readthedocs.org/projects/xsimd/badge/?version=latest)](https://xsimd.readthedocs.io/en/latest/?badge=latest)
[![Join the Gitter Chat](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/QuantStack/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

C++ wrappers for SIMD intrinsics

## Introduction

Find out more about this implementation of C++ wrappers for SIMD intrinsics at the [The C++ Scientist](http://johanmabille.github.io/blog/archives/).

## Installation

`xsimd` is a header-only library. We provide a package for the conda package manager.

```bash
conda install -c conda-forge xsimd
```

Or you can directly install it from the sources:

```bash
cmake -D CMAKE_INSTALL_PREFIX=your_install_prefix
make install
```

## Building and Running the Tests

Building the tests requires the [GTest](https://github.com/google/googletest) testing framework and [cmake](https://cmake.org).

gtest and cmake are available as a packages for most linux distributions. Besides, they can also be installed with the `conda` package manager (even on windows):

```bash
conda install -c conda-forge gtest cmake
```

Once `gtest` and `cmake` are installed, you can build and run the tests:

```bash
mkdir build
cd build
cmake ../
make xtest
```

In the context of continuous integration with Travis CI, tests are run in a `conda` environment, which can be activated with

```bash
cd test
conda env create -f ./test-environment.yml
source activate test-xsimd
cd ..
cmake .
make xtest
```

## Building the HTML Documentation

xsimd's documentation is built with three tools

 - [doxygen](http://www.doxygen.org)
 - [sphinx](http://www.sphinx-doc.org)
 - [breathe](https://breathe.readthedocs.io)

While doxygen must be installed separately, you can install breathe by typing

```bash
pip install breathe
``` 

Breathe can also be installed with `conda`

```bash
conda install -c conda-forge breathe
```

Finally, build the documentation with

```bash
make html
```

from the `docs` subdirectory.

## License

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions.

This software is licensed under the BSD-3-Clause license. See the [LICENSE](LICENSE) file for details.

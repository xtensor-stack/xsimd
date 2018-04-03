# ![xsimd](http://quantstack.net/assets/images/xsimd.svg)

[![Travis](https://travis-ci.org/QuantStack/xsimd.svg?branch=master)](https://travis-ci.org/QuantStack/xsimd)
[![Appveyor](https://ci.appveyor.com/api/projects/status/unk39xhu0wxmiif7?svg=true)](https://ci.appveyor.com/project/QuantStack/xsimd)
[![Documentation Status](http://readthedocs.org/projects/xsimd/badge/?version=latest)](https://xsimd.readthedocs.io/en/latest/?badge=latest)
[![Join the Gitter Chat](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/QuantStack/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

C++ wrappers for SIMD intrinsics

## Introduction

SIMD (Single Instruction, Multiple Data) is a feature of microprocessors that has been available for many years. SIMD instructions perform a single operation
on a batch of values at once, and thus provide a way to significantly accelerate code execution. However, these instructions differ between microprocessor
vendors and compilers.

`xsimd` provides a unified means for using these features for library authors. Namely, it enables manipulation of batches of numbers with the same arithmetic
operators as for single values. It also provides accelerated implementation of common mathematical functions operating on batches.

You can find out more about this implementation of C++ wrappers for SIMD intrinsics at the [The C++ Scientist](http://johanmabille.github.io/blog/archives/).
The mathematical functions are a lightweight implementation of the algorithms used in [boost.SIMD](https://github.com/NumScale/boost.simd).

`xsimd` requires a C++14 compliant compiler. The following C++ compilers are supported:

Compiler                | Version
------------------------|-------------------------------
Microsoft Visual Studio | MSVC 2015 update 2 and above
g++                     | 4.9 and above
clang                   | 3.7 and above

The following SIMD instruction set extensions are supported:

Architecture | Instruction set extensions
-------------|-----------------------------------------------------
x86          | SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, FMA3, AVX2
x86 AMD      | same as above + SSE4A, FMA4, XOP
ARM          | ARMv7, ARMv8

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

## Usage

### Explicit use of an instruction set extension

Here is an example that computes the mean of two sets of 4 double floating point values, assuming AVX extension is supported:
```cpp
#include <iostream>
#include "xsimd/xsimd.hpp"

namespace xs = xsimd;

int main(int argc, char* argv[])
{
    xs::batch<double, 4> a(1.5, 2.5, 3.5, 4.5);
    xs::batch<double, 4> b(2.5, 3.5, 4.5, 5.5);
    auto mean = (a + b) / 2;
    std::cout << mean << std::endl;
    return 0;
}
```

This example outputs:

```cpp
(2.0, 3.0, 4.0, 5.0)
```

### Auto detection of the instruction set extension to be used

The same computation operating on vectors and using the most performant instruction set available:

```cpp
#include <cstddef>
#include <vector>
#include "xsimd/xsimd.hpp"

namespace xs = xsimd;
using vector_type = std::vector<double, xsimd::aligned_allocator<double, XSIMD_DEFAULT_ALIGNMENT>>;

void mean(const vector_type& a, const vector_type& b, vector_type& res)
{
    std::size_t size = a.size();
    constexpr std::size_t simd_size = xsimd::simd_type<double>::size;
    std::size_t vec_size = size - size % simd_size;

    for(std::size_t i = 0; i < vec_size; i += simd_size)
    {
        auto ba = xs::load_aligned(&a[i]);
        auto bb = xs::load_aligned(&b[i]);
        auto bres = (ba + bb) / 2;
        bres.store_aligned(&res[i]);
    }
    for(std::size_t i = vec_size; i < size; ++i)
    {
        res[i] = (a[i] + b[i]) / 2;
    }
}
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

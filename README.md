# ![xsimd](docs/source/xsimd.svg)

[![GHA android](https://github.com/xtensor-stack/xsimd/actions/workflows/android.yml/badge.svg)](https://github.com/xtensor-stack/xsimd/actions/workflows/android.yml)
[![GHA cross-rvv](https://github.com/xtensor-stack/xsimd/actions/workflows/cross-rvv.yml/badge.svg)](https://github.com/xtensor-stack/xsimd/actions/workflows/cross-rvv.yml)
[![GHA cross-sve](https://github.com/xtensor-stack/xsimd/actions/workflows/cross-sve.yml/badge.svg)](https://github.com/xtensor-stack/xsimd/actions/workflows/cross-sve.yml)
[![GHA cross](https://github.com/xtensor-stack/xsimd/actions/workflows/cross.yml/badge.svg)](https://github.com/xtensor-stack/xsimd/actions/workflows/cross.yml)
[![GHA cxx-no-exceptions](https://github.com/xtensor-stack/xsimd/actions/workflows/cxx-no-exceptions.yml/badge.svg)](https://github.com/xtensor-stack/xsimd/actions/workflows/cxx-no-exceptions.yml)
[![GHA cxx-versions](https://github.com/xtensor-stack/xsimd/actions/workflows/cxx-versions.yml/badge.svg)](https://github.com/xtensor-stack/xsimd/actions/workflows/cxx-versions.yml)
[![GHA emscripten](https://github.com/xtensor-stack/xsimd/actions/workflows/emscripten.yml/badge.svg)](https://github.com/xtensor-stack/xsimd/actions/workflows/emscripten.yml)
[![GHA linux](https://github.com/xtensor-stack/xsimd/actions/workflows/linux.yml/badge.svg)](https://github.com/xtensor-stack/xsimd/actions/workflows/linux.yml)
[![GHA macos](https://github.com/xtensor-stack/xsimd/actions/workflows/macos.yml/badge.svg)](https://github.com/xtensor-stack/xsimd/actions/workflows/macos.yml)
[![GHA windows](https://github.com/xtensor-stack/xsimd/actions/workflows/windows.yml/badge.svg)](https://github.com/xtensor-stack/xsimd/actions/workflows/windows.yml)
[![Documentation Status](http://readthedocs.org/projects/xsimd/badge/?version=latest)](https://xsimd.readthedocs.io/en/latest/?badge=latest)
[![Join the Gitter Chat](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/QuantStack/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

C++ wrappers for SIMD intrinsics

## Introduction

SIMD (Single Instruction, Multiple Data) is a feature of microprocessors that has been available for many years. SIMD instructions perform a single operation
on a batch of values at once, and thus provide a way to significantly accelerate code execution. However, these instructions differ between microprocessor
vendors and compilers.

`xsimd` provides a unified means for using these features for library authors. Namely, it enables manipulation of batches of numbers with the same arithmetic operators as for single values. It also provides accelerated implementation of common mathematical functions operating on batches.

## Adoption

Beyond Xtensor, Xsimd has been adopted by major open-source projects, such as Mozilla Firefox, Apache Arrow, Pythran, and Krita.

## History

The XSimd project started with a series of blog articles by Johan Mabille on how to implement wrappers for SIMD intrinsicts.
The archives of the blog can be found here: [The C++ Scientist](http://johanmabille.github.io/blog/archives/). The design described in
the articles remained close to the actual architecture of XSimd up until Version 8.0.

The mathematical functions are a lightweight implementation of the algorithms originally implemented in the now deprecated [boost.SIMD](https://github.com/NumScale/boost.simd) project.

## Requirements

`xsimd` requires a C++11 compliant compiler. The following C++ compilers are supported:

Compiler                | Version
------------------------|-------------------------------
Microsoft Visual Studio | MSVC 2015 update 2 and above
g++                     | 4.9 and above
clang                   | 4.0 and above

The following SIMD instruction set extensions are supported:

Architecture | Instruction set extensions
-------------|-----------------------------------------------------
x86          | SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, FMA3+SSE, FMA3+AVX, FMA3+AVX2
x86          | AVX512BW, AVX512CD, AVX512DQ, AVX512F (gcc7 and higher)
x86 AMD      | FMA4
ARM          | NEON, NEON64, SVE128/256/512 (fixed vector size)
WebAssembly  | WASM
RISC-V       | RISC-V128/256/512 (fixed vector size)

## Installation

### Install from conda-forge

A package for xsimd is available on the mamba (or conda) package manager.

```bash
mamba install -c conda-forge xsimd
```

### Install with Spack

A package for xsimd is available on the Spack package manager.

```bash
spack install xsimd
spack load xsimd
```

### Install from sources

You can directly install it from the sources with cmake:

```bash
cmake -D CMAKE_INSTALL_PREFIX=your_install_prefix
make install
```

## Documentation

To get started with using `xsimd`, check out the full documentation

http://xsimd.readthedocs.io/

## Dependencies

`xsimd` has an optional dependency on the [xtl](https://github.com/xtensor-stack/xtl) library:

| `xsimd` | `xtl` (optional) |
|---------|------------------|
|  master |     ^0.7.0       |
|  12.x   |     ^0.7.0       |
|  11.x   |     ^0.7.0       |
|  10.x   |     ^0.7.0       |
|   9.x   |     ^0.7.0       |
|   8.x   |     ^0.7.0       |

The dependency on `xtl` is required if you want to support vectorization for `xtl::xcomplex`. In this case, you must build your project with C++14 support enabled.

## Usage

The version 8 of the library is a complete rewrite and there are some slight differences with 7.x versions.
A migration guide will be available soon. In the meanwhile, the following examples show how to use both versions
7 and 8 of the library?

### Explicit use of an instruction set extension

Here is an example that computes the mean of two sets of 4 double floating point values, assuming AVX extension is supported:
```cpp
#include <iostream>
#include "xsimd/xsimd.hpp"

namespace xs = xsimd;

int main(int argc, char* argv[])
{
    xs::batch<double, xs::avx2> a = {1.5, 2.5, 3.5, 4.5};
    xs::batch<double, xs::avx2> b = {2.5, 3.5, 4.5, 5.5};
    auto mean = (a + b) / 2;
    std::cout << mean << std::endl;
    return 0;
}
```

Do not forget to enable AVX extension when building the example. With gcc or clang, this is done with the `-mavx` flag,
on MSVC you have to pass the `/arch:AVX` option.

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
using vector_type = std::vector<double, xsimd::aligned_allocator<double>>;

void mean(const vector_type& a, const vector_type& b, vector_type& res)
{
    std::size_t size = a.size();
    constexpr std::size_t simd_size = xsimd::simd_type<double>::size;
    std::size_t vec_size = size - size % simd_size;

    for(std::size_t i = 0; i < vec_size; i += simd_size)
    {
        auto ba = xs::load_aligned(&a[i]);
        auto bb = xs::load_aligned(&b[i]);
        auto bres = (ba + bb) / 2.;
        bres.store_aligned(&res[i]);
    }
    for(std::size_t i = vec_size; i < size; ++i)
    {
        res[i] = (a[i] + b[i]) / 2.;
    }
}
```

## Building and Running the Tests

Building the tests requires [cmake](https://cmake.org).

`cmake` is available as a package for most linux distributions. Besides, they can also be installed with the `conda` package manager (even on windows):

```bash
conda install -c conda-forge cmake
```

Once `cmake` is installed, you can build and run the tests:

```bash
mkdir build
cd build
cmake ../ -DBUILD_TESTS=ON
make xtest
```

In the context of continuous integration with Travis CI, tests are run in a `conda` environment, which can be activated with

```bash
cd test
conda env create -f ./test-environment.yml
source activate test-xsimd
cd ..
cmake . -DBUILD_TESTS=ON
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

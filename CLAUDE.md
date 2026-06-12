# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

xsimd is a header-only C++ library providing wrappers for SIMD (Single Instruction, Multiple Data) intrinsics. It enables writing vectorized code with a unified API that works across multiple CPU architectures (x86, ARM, PowerPC, RISC-V, WebAssembly). The library is used by major projects including Mozilla Firefox, Apache Arrow, Pythran, and Krita.

**Key characteristics:**
- Header-only library (no compilation needed)
- Requires C++14 minimum (C++17 if using xtl complex support)
- Architecture detection at compile-time
- Mathematical functions implemented similar to boost.SIMD

## Build System

### Using Pixi (Recommended for Development)

Pixi manages conda environments and build tasks. Common commands:

```bash
# Run tests with default environment and preset
pixi run test

# Run tests with specific compiler environment
pixi run -e clang-21 test
pixi run -e clang-18 test
pixi run -e gcc-15 test

# Run tests targeting specific SIMD architecture
pixi run -e clang-21 test dev-sse2
pixi run -e clang-21 test dev-avx2
pixi run -e clang-21 test dev-neon

# Format code
pixi run fmt

# Build documentation
pixi run doc

# Initialize LSP/clangd compilation database
pixi run init-lsp
```

### Using CMake Directly

```bash
# Configure and build tests (basic)
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make xtest

# Configure with specific preset
cmake -B build/my-build --preset dev-native -G Ninja
cmake --build build/my-build --parallel

# Run tests
./build/my-build/test/test_xsimd

# Build with XTL complex support
cmake .. -DENABLE_XTL_COMPLEX=ON -DBUILD_TESTS=ON

# Build benchmarks
cmake .. -DBUILD_BENCHMARK=ON

# Build examples
cmake .. -DBUILD_EXAMPLES=ON
```

### CMake Presets

The repository includes CMake presets for testing different SIMD instruction sets:
- `dev-native`: Uses -march=native
- `dev-sse2`, `dev-sse3`, `dev-ssse3`, `dev-sse4.1`, `dev-sse4.2`: x86 SSE variants
- `dev-avx`, `dev-avx2`: x86 AVX variants
- `dev-neon`: ARM NEON
- `dev-sve`: ARM SVE

## Code Architecture

### Directory Structure

```
include/xsimd/
├── arch/              # Architecture-specific implementations
│   ├── common/        # Shared implementations (arithmetic, math, swizzle, etc.)
│   ├── xsimd_*.hpp    # Per-architecture implementations (sse2, avx2, neon, etc.)
│   └── xsimd_isa.hpp  # Instruction set detection
├── config/            # Configuration and CPU detection
│   ├── xsimd_arch.hpp # Architecture list and selection logic
│   └── xsimd_cpuid.hpp # Runtime CPU feature detection
├── types/             # Core batch types
│   ├── xsimd_batch.hpp          # Main batch<T,A> type
│   ├── xsimd_batch_constant.hpp # Compile-time constant batches
│   ├── xsimd_*_register.hpp     # Per-architecture register wrappers
│   └── xsimd_api.hpp            # High-level API functions
├── memory/            # Memory utilities
└── xsimd.hpp          # Main entry point

test/                  # Test suite (uses doctest)
benchmark/             # Performance benchmarks
examples/              # Usage examples
docs/                  # Sphinx documentation
```

### Core Concepts

1. **batch<T, A>**: The primary type representing a SIMD register
   - `T`: scalar type (float, double, int32_t, etc.)
   - `A`: architecture (avx2, neon64, default_arch, etc.)
   - Provides operator overloads (+, -, *, /, etc.)

2. **Architecture Abstraction**:
   - Each architecture (SSE2, AVX2, NEON, etc.) has its own implementation files
   - `arch/common/` contains fallback implementations
   - Architecture selection happens via template specialization

3. **API Layers**:
   - `xsimd_batch.hpp`: Batch type definition and operators
   - `arch/xsimd_*.hpp`: Architecture-specific kernel implementations
   - `types/xsimd_api.hpp`: High-level functions (abs, sqrt, exp, etc.)

4. **Architecture Detection**:
   - Compile-time: Based on compiler flags (-mavx2, -march=native)
   - Runtime: CPUID detection available in `config/xsimd_cpuid.hpp`
   - `default_arch`: Automatically selects best available architecture

### Key Design Patterns

- **Header-only**: All code is in headers; implementations use inline functions
- **SFINAE/Enable-if**: Used extensively for type traits and architecture selection
- **Tag Dispatching**: Architecture types used as tags for function overloading
- **Batch Operations**: All operations return new batches (immutable style)

## Development Workflow

### Adding New Functionality

1. Add kernel implementation in relevant `arch/` files (start with `arch/common/`)
2. Add specializations for specific architectures if needed
3. Add tests in `test/test_*.cpp` (must work on all architectures)
4. Add documentation in `docs/source/api/`

### Testing Specific Architectures

Use CMake presets to test older instruction sets on newer machines:
```bash
pixi run -e clang-21 test dev-sse2  # Test SSE2 on AVX2+ machine
```

### Code Formatting

The project uses clang-format. Format files before committing:
```bash
pixi run fmt
# or manually:
find . -name '*.[ch]pp' | xargs clang-format -i
```

### Running Single Tests

The test suite uses doctest. To run specific tests:
```bash
# Build tests
cmake --build build/my-build

# Run all tests
./build/my-build/test/test_xsimd

# Run tests matching a pattern
./build/my-build/test/test_xsimd --test-case="*batch*"

# List all test cases
./build/my-build/test/test_xsimd --list-test-cases
```

## Important Files

- `include/xsimd/xsimd.hpp`: Main header users include
- `include/xsimd/config/xsimd_arch.hpp`: Architecture detection and selection
- `include/xsimd/types/xsimd_batch.hpp`: Core batch type
- `include/xsimd/arch/xsimd_isa.hpp`: Supported instruction sets
- `test/CMakeLists.txt`: Test configuration (architecture flags, cross-compilation)

## Cross-Compilation and CI

- ARM cross-compilation uses QEMU (see `test/CMakeLists.txt`)
- CI tests multiple architectures, compilers, and platforms
- Emscripten support for WebAssembly SIMD

## Notes

- When modifying architecture-specific code, ensure changes work across all supported architectures
- The library is header-only, so implementation changes affect ABI
- Version 8.0 was a major rewrite; older code may follow different patterns
- Mathematical functions are inspired by the deprecated boost.SIMD library

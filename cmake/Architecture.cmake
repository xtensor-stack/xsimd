include(CheckCXXCompilerFlag)

# Get the target architecture family.
function(xsimd_get_target out_var)
    if(MSVC AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID)
        set(proc "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}")
    else()
        set(proc "${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    string(TOLOWER "${proc}" proc)

    # The processor name is the host one when cross-compiling with -m32.
    if(proc MATCHES "^(aarch64|arm)")
        if(CMAKE_SIZEOF_VOID_P EQUAL 4)
            set(${out_var} "arm32" PARENT_SCOPE)
        else()
            set(${out_var} "arm64" PARENT_SCOPE)
        endif()
    elseif(proc MATCHES "^riscv64")
        set(${out_var} "riscv64" PARENT_SCOPE)
    elseif(proc MATCHES "^(x86|x64|x86_64|amd64|i[3-6]86)$")
        if(CMAKE_SIZEOF_VOID_P EQUAL 4)
            set(${out_var} "x86_32" PARENT_SCOPE)
        else()
            set(${out_var} "x86_64" PARENT_SCOPE)
        endif()
    else()
        set(${out_var} "" PARENT_SCOPE)
    endif()
endfunction()


# Get xsimd flags for emulated architecture
function(xsimd_get_emulated_arch_flags out_list arch)
    if(arch MATCHES "emulated[-_<]([0-9]+)>?")
        set(
            ${out_list}
            "-DXSIMD_DEFAULT_ARCH=emulated<${CMAKE_MATCH_1}>;-DXSIMD_WITH_EMULATED=1"
            PARENT_SCOPE
        )
    else()
        message(FATAL_ERROR "Unknown xsimd emulated architecture: ${arch}")
    endif()
endfunction()


# Get flags for native build.
function(xsimd_get_native_arch_flags out_list)
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        message(WARNING "Native not supported on MSVC, not setting any flag")
    else()
        set(${out_list} "-march=native" PARENT_SCOPE)
    endif()
endfunction()


function(xsimd_get_arm64_arch_flags_unix out_list arch arm64_baseline)
    if(arch MATCHES "sve[-_]([0-9]+)")
        if(arm64_baseline STREQUAL "")
            # Starting point for SVE
            set(arm64_baseline "armv8.2-a")
        endif()
        set(
            ${out_list}
            "-march=${arm64_baseline}+sve;-msve-vector-bits=${CMAKE_MATCH_1}"
            PARENT_SCOPE
        )
    elseif(arch MATCHES "neon")
        if(arm64_baseline STREQUAL "")
            set(arm64_baseline "armv8-a")
        endif()
        set(${out_list} "-march=${arm64_baseline}" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Unknown xsimd architecture for arm64: ${arch}")
    endif()
endfunction()


function(xsimd_get_arm64_arch_flags_msvc out_list arch arm64_baseline)
    if(arch MATCHES "sve")
        message(FATAL_ERROR "SVE not supported on MSVC compilers")
    elseif(arch MATCHES "neon")
        if(arm64_baseline STREQUAL "")
            set(arm64_baseline "armv8.0")
        endif()
        set(${out_list} "/arch:${arm64_baseline}" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Unknown xsimd architecture for arm64: ${arch}")
    endif()
endfunction()


# Get the flag to compile with the desired xsimd arch on arm64.
function(xsimd_get_arm64_arch_flags out_list arch arm64_baseline)
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        xsimd_get_arm64_arch_flags_msvc(flags "${arch}" "${arm64_baseline}")
    else()
        xsimd_get_arm64_arch_flags_unix(flags "${arch}" "${arm64_baseline}")
    endif()
    set(${out_list} ${flags} PARENT_SCOPE)
endfunction()


# Get the flag to compile with the desired xsimd arch on riscv64.
function(xsimd_get_riscv64_arch_flags out_list arch riscv64_baseline)
    if(NOT arch MATCHES "rvv[-_]([0-9]+)")
        message(FATAL_ERROR "Unknown xsimd architecture for riscv64: ${arch}")
    endif()
    set(vector_bits "${CMAKE_MATCH_1}")
    if(riscv64_baseline STREQUAL "")
        set(riscv64_baseline "rv64gc")
    endif()
    # Single-letter extensions must precede the multi-letter ones.
    string(REGEX MATCH "^[^_]*" base "${riscv64_baseline}")
    string(REGEX MATCH "_.*" extensions "${riscv64_baseline}")
    set(
        ${out_list}
        "-march=${base}v${extensions}_zvl${vector_bits}b;-mrvv-vector-bits=zvl"
        PARENT_SCOPE
    )
endfunction()


# Get the flag to compile with the desired xsimd arch on arm32.
function(xsimd_get_arm32_arch_flags out_list arch arm32_baseline)
    if(NOT arch MATCHES "neon")
        message(FATAL_ERROR "Unknown xsimd architecture for arm32: ${arch}")
    endif()

    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        # Neon is mandatory on Windows on ARM.
        if(NOT arm32_baseline STREQUAL "")
            message(WARNING "ARM32_BASELINE is ignored on MSVC: ${arm32_baseline}")
        endif()
    else()
        if(arm32_baseline STREQUAL "")
            set(arm32_baseline "armv7-a")
        endif()
        # -mfloat-abi is left to the toolchain as it must match the sysroot ABI.
        set(${out_list} "-march=${arm32_baseline};-mfpu=neon" PARENT_SCOPE)
    endif()
endfunction()


function(xsimd_get_x86_arch_flags_unix out_list arch)
    if(arch STREQUAL "sse2")
        set(flags "-msse2;-mno-sse3")
    elseif(arch STREQUAL "sse3")
        set(flags "-msse3;-mno-ssse3")
    elseif(arch STREQUAL "ssse3")
        set(flags "-mssse3;-mno-sse4.1")
    elseif(arch STREQUAL "sse4.1")
        set(flags "-msse4.1;-mno-sse4.2")
    elseif(arch STREQUAL "sse4.2")
        set(flags "-msse4.2;-mno-avx")
    elseif(arch STREQUAL "avx")
        set(flags "-mavx;-mno-avx2")
    elseif(arch MATCHES "avx[_-]128")
        set(flags "-mavx;-mno-avx2;-DXSIMD_DEFAULT_ARCH=${arch}")
    elseif(arch STREQUAL "avx2")
        set(flags "-mavx2;-mno-avx512f")
    elseif(arch MATCHES "avx2[_-]128")
        set(flags  "-mavx2;-mno-avx512f;-DXSIMD_DEFAULT_ARCH=${arch}")
    elseif(arch MATCHES "^avx512")
        set(flags "-mavx;-mavx2;-mavx512f")
        # Extensions not enabled are explicitly disabled.
        set(avx512_extensions cd dq bw er pf ifma vbmi vbmi2 vnni)
        if(arch STREQUAL "avx512f")
            set(enabled)
        elseif(arch STREQUAL "avx512cd")
            set(enabled cd)
        elseif(arch STREQUAL "avx512dq")
            set(enabled cd dq)
        elseif(arch STREQUAL "avx512bw")
            set(enabled cd dq bw)
        elseif(arch STREQUAL "avx512er")
            set(enabled cd dq er)
        elseif(arch STREQUAL "avx512pf")
            set(enabled cd dq er pf)
        elseif(arch STREQUAL "avx512ifma")
            set(enabled cd dq bw ifma)
        elseif(arch STREQUAL "avx512vbmi")
            set(enabled cd dq bw ifma vbmi)
        elseif(arch STREQUAL "avx512vbmi2")
            set(enabled cd dq bw ifma vbmi vbmi2)
        elseif(arch STREQUAL "avx512vnni_avx512bw")
            set(enabled cd dq bw vnni)
        elseif(arch STREQUAL "avx512vnni_avx512vbmi2")
            set(enabled cd dq bw ifma vbmi vbmi2 vnni)
        elseif(arch MATCHES "^avx512vl[_-](128|256)$")
            set(enabled cd dq bw)
            list(APPEND flags "-mavx512vl" "-DXSIMD_DEFAULT_ARCH=${arch}")
        else()
            message(FATAL_ERROR "Unknown xsimd architecture for x86: ${arch}")
        endif()
        foreach(ext IN LISTS avx512_extensions)
            if(ext IN_LIST enabled)
                list(APPEND flags "-mavx512${ext}")
            else()
                list(APPEND flags "-mno-avx512${ext}")
            endif()
        endforeach()
    else()
        message(FATAL_ERROR "Unknown xsimd architecture for x86: ${arch}")
    endif()

    set(${out_list} ${flags} PARENT_SCOPE)
endfunction()


function(xsimd_get_x86_arch_flags_msvc out_list arch sse2_flags)
    set(mismatch)
    if(arch STREQUAL "sse2")
        set(flags ${sse2_flags})
    elseif(arch MATCHES "^(sse3|ssse3|sse4\\.1)$")
        set(flags ${sse2_flags})
        set(mismatch "no /arch flag for ${arch}, falling back to SSE2")
    elseif(arch STREQUAL "sse4.2")
        set(flags "/arch:SSE4.2")
    elseif(arch STREQUAL "avx")
        set(flags "/arch:AVX")
    elseif(arch STREQUAL "avx_128")
        set(flags "/arch:AVX;-DXSIMD_DEFAULT_ARCH=${arch}")
    elseif(arch STREQUAL "avx2")
        set(flags "/arch:AVX2")
        set(mismatch "/arch:AVX2 also enables FMA3")
    elseif(arch STREQUAL "avx2_128")
        set(flags "/arch:AVX2;-DXSIMD_DEFAULT_ARCH=${arch}")
        set(mismatch "/arch:AVX2 also enables FMA3")
    elseif(arch MATCHES "^avx512vl_(128|256)$")
        set(flags "/arch:AVX512;-DXSIMD_DEFAULT_ARCH=${arch}")
        set(mismatch "/arch:AVX512 also enables FMA3")
    elseif(arch MATCHES "^avx512(f|cd|dq|bw)$")
        set(flags "/arch:AVX512")
        set(mismatch "/arch:AVX512 enables the full AVX512F/CD/DQ/BW/VL set and FMA3")
    elseif(arch MATCHES "^avx512(er|pf|ifma|vbmi|vbmi2|vnni_avx512bw|vnni_avx512vbmi2)$")
        set(flags "/arch:AVX512")
        set(mismatch "no /arch flag for ${arch}, falling back to AVX512F/CD/DQ/BW/VL")
    else()
        message(FATAL_ERROR "Unknown xsimd architecture for x86: ${arch}")
    endif()

    if(mismatch)
        message(WARNING "MSVC flags do not exactly match xsimd architecture ${arch}: ${mismatch}")
    endif()

    set(${out_list} ${flags} PARENT_SCOPE)
endfunction()


function(xsimd_get_x86_32_arch_flags_unix out_list arch x86_32_baseline)
    xsimd_get_x86_arch_flags_unix(flags "${arch}")
    if(NOT x86_32_baseline STREQUAL "")
        list(PREPEND flags "-march=${x86_32_baseline}")
    endif()
    set(${out_list} ${flags} PARENT_SCOPE)
endfunction()


function(xsimd_get_x86_32_arch_flags_msvc out_list arch x86_32_baseline)
    if(NOT x86_32_baseline STREQUAL "")
        message(WARNING "X86_32_BASELINE is ignored on MSVC: ${x86_32_baseline}")
    endif()
    xsimd_get_x86_arch_flags_msvc(flags "${arch}" "/arch:SSE2")
    set(${out_list} ${flags} PARENT_SCOPE)
endfunction()


function(xsimd_get_x86_32_arch_flags out_list arch x86_32_baseline)
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        xsimd_get_x86_32_arch_flags_msvc(flags "${arch}" "${x86_32_baseline}")
    else()
        xsimd_get_x86_32_arch_flags_unix(flags "${arch}" "${x86_32_baseline}")
    endif()
    set(${out_list} ${flags} PARENT_SCOPE)
endfunction()


function(xsimd_get_x86_64_arch_flags_unix out_list arch x86_64_baseline)
    xsimd_get_x86_arch_flags_unix(flags "${arch}")
    # Default to the microarchitecture level just below the requested arch.
    if(x86_64_baseline STREQUAL "")
        if(arch MATCHES "^avx512")
            set(x86_64_baseline "x86-64-v3")
        elseif(arch MATCHES "^avx")
            set(x86_64_baseline "x86-64-v2")
        elseif(arch MATCHES "^(sse3|ssse3|sse4)")
            set(x86_64_baseline "x86-64")
        endif()
    endif()
    if(NOT x86_64_baseline STREQUAL "")
        list(PREPEND flags "-march=${x86_64_baseline}")
    endif()
    set(${out_list} ${flags} PARENT_SCOPE)
endfunction()


function(xsimd_get_x86_64_arch_flags_msvc out_list arch x86_64_baseline)
    if(NOT x86_64_baseline STREQUAL "")
        message(WARNING "X86_64_BASELINE is ignored on MSVC: ${x86_64_baseline}")
    endif()
    # SSE2 is the x64 baseline, /arch:SSE2 does not exist there.
    xsimd_get_x86_arch_flags_msvc(flags "${arch}" "")
    set(${out_list} ${flags} PARENT_SCOPE)
endfunction()


function(xsimd_get_x86_64_arch_flags out_list arch x86_64_baseline)
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        xsimd_get_x86_64_arch_flags_msvc(flags "${arch}" "${x86_64_baseline}")
    else()
        xsimd_get_x86_64_arch_flags_unix(flags "${arch}" "${x86_64_baseline}")
    endif()
    set(${out_list} ${flags} PARENT_SCOPE)
endfunction()


function(xsimd_target_set_arch target scope)
    # Names of option parameters (without arguments)
    set(options)
    # Names of named parameters with a single argument
    set(one_value_args ARCH ARM32_BASELINE ARM64_BASELINE RISCV64_BASELINE X86_32_BASELINE X86_64_BASELINE)
    # Names of named parameters with a multiple arguments
    set(multi_values_args)
    cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_values_args}" ${ARGN})
    if(ARG_UNPARSED_ARGUMENTS)
        message(
            AUTHOR_WARNING
            "Unrecognized options passed to ${CMAKE_CURRENT_FUNCTION}: "
            "${ARG_UNPARSED_ARGUMENTS}"
        )
    endif()

    if(NOT scope STREQUAL "PUBLIC" AND NOT scope STREQUAL "PRIVATE" AND NOT scope STREQUAL "INTERFACE")
        message(FATAL_ERROR "scope must be PUBLIC, PRIVATE, or INTERFACE, got: ${scope}")
    endif()

    if(NOT ARG_ARCH)
        message(FATAL_ERROR "Select an xsimd architecture")
    endif()
    string(TOLOWER "${ARG_ARCH}" arch)

    xsimd_get_target(target_arch)
    if(arch STREQUAL "native")
        xsimd_get_native_arch_flags(flags)
    elseif(arch STREQUAL "generic")
        set(flags)
    elseif(arch MATCHES "emulated")
        xsimd_get_emulated_arch_flags(flags "${arch}")
    elseif(target_arch STREQUAL "arm32")
        xsimd_get_arm32_arch_flags(flags "${arch}" "${ARG_ARM32_BASELINE}")
    elseif(target_arch STREQUAL "arm64")
        xsimd_get_arm64_arch_flags(flags "${arch}" "${ARG_ARM64_BASELINE}")
    elseif(target_arch STREQUAL "riscv64")
        xsimd_get_riscv64_arch_flags(flags "${arch}" "${ARG_RISCV64_BASELINE}")
    elseif(target_arch STREQUAL "x86_32")
        xsimd_get_x86_32_arch_flags(flags "${arch}" "${ARG_X86_32_BASELINE}")
    elseif(target_arch STREQUAL "x86_64")
        xsimd_get_x86_64_arch_flags(flags "${arch}" "${ARG_X86_64_BASELINE}")
    endif()

    target_compile_options(${target} ${scope} ${flags})
endfunction()


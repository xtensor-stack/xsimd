include(CheckCXXCompilerFlag)


function(xsimd_normalize_arch out_var arch)
    string(TOUPPER "${arch}" arch)
    string(REGEX REPLACE "[._\-]" "" arch "${arch}")
    set(${out_var} "${arch}" PARENT_SCOPE)
endfunction()


#############################
#  x86 micro architectures  #
#############################

if(MSVC)
    # No fine-grain compilation target on MSVC for x86
    # https://learn.microsoft.com/en-us/cpp/build/reference/arch-x64
    # We prefer to omit the micro architecture so that the user has to
    # epxlicitly select a different one.
    set(XSIMD_SSE2_COMPILE_ARGS "/arch:SSE2")
    set(XSIMD_SSE4_2_COMPILE_ARGS "/arch:SSE4.2")
    set(XSIMD_AVX_COMPILE_ARGS "/arch:AVX")
    set(XSIMD_AVX2_COMPILE_ARGS "/arch:AVX2")
    set(XSIMD_AVX512DQ_COMPILE_ARGS "/arch:AVX512")
else()
    check_cxx_compiler_flag("-mno-avx512f" XSIMD_HAS_MNO_AVX512F)
    if(XSIMD_HAS_MNO_AVX512F)
        set(XSIMD_NOAVX512F_COMPILE_ARGS "-mno-avx512f")
    else()
        set(XSIMD_NOAVX512F_COMPILE_ARGS)
    endif()

    check_cxx_compiler_flag("-mno-avx10.1" XSIMD_HAS_MNO_AVX10)
    if(XSIMD_HAS_MNO_AVX10)
        set(XSIMD_NOAVX10_COMPILE_ARGS "-mno-avx10.1")
    else()
        set(XSIMD_NOAVX10_COMPILE_ARGS)
    endif()

    set(XSIMD_SSE2_COMPILE_ARGS "-msse2" "-mno-sse3")
    set(XSIMD_SSE3_COMPILE_ARGS "-msse3" "-mno-ssse3")
    set(XSIMD_SSSE3_COMPILE_ARGS "-mssse3" "-mno-sse4.1")
    set(XSIMD_SSE4_1_COMPILE_ARGS "-msse4.1" "-mno-sse4.2")
    set(XSIMD_SSE4_2_COMPILE_ARGS "-msse4.2" "-mno-avx")
    set(XSIMD_AVX_COMPILE_ARGS "-mavx" "-mno-avx2")
    set(XSIMD_AVX2_COMPILE_ARGS "-mavx2" ${XSIMD_NOAVX512F_COMPILE_ARGS})
    set(XSIMD_AVX512F_COMPILE_ARGS "-mavx512f" ${XSIMD_NOAVX10_COMPILE_ARGS})
    set(XSIMD_AVX512DQ_COMPILE_ARGS "-mavx512dq" ${XSIMD_NOAVX10_COMPILE_ARGS})
endif()

function(xsimd_x86_compile_args out_var arch)
    xsimd_normalize_arch(arch "${arch}")
    if(NOT DEFINED XSIMD_${arch}_COMPILE_ARGS)
        message(FATAL_ERROR "Unknown xsimd architecture: ${arch}")
    endif()
    set(${out_var} "${XSIMD_${arch}_COMPILE_ARGS}" PARENT_SCOPE)
endfunction()

#############################
#  Arm micro architectures  #
#############################

function(get_effective_march target result_var)
    set(march "")

    # Helper macro to extract last -march from a flags string
    macro(extract_march_from_string flags)
        # Must loop as string may have multiple -march, last wins
        string(REGEX MATCHALL "-march=[^ ]+" matches "${flags}")
        if(matches)
            list(GET matches -1 last_match)
            string(REGEX REPLACE "-march=" "" march "${last_match}")
        endif()
    endmacro()

    # Helper macro to extract last -march from a list of options
    macro(extract_march_from_list opts)
        foreach(opt IN LISTS opts)
            if(opt MATCHES "^-march=(.+)$")
                set(march ${CMAKE_MATCH_1})
            endif()
        endforeach()
    endmacro()

    # 1. CMAKE_CXX_FLAGS (lowest priority)
    extract_march_from_string("${CMAKE_CXX_FLAGS}")

    # 2. CMAKE_CXX_FLAGS_<CONFIG>
    string(TOUPPER "${CMAKE_BUILD_TYPE}" config_upper)
    if(config_upper)
        extract_march_from_string("${CMAKE_CXX_FLAGS_${config_upper}}")
    endif()

    # 3. Directory COMPILE_OPTIONS (target's own directory if possible)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.19")
        get_target_property(target_dir ${target} SOURCE_DIR)
        get_directory_property(dir_opts DIRECTORY "${target_dir}" COMPILE_OPTIONS)
    else()
        get_directory_property(dir_opts COMPILE_OPTIONS)  # best effort fallback
    endif()
    if(dir_opts)
        extract_march_from_list("${dir_opts}")
    endif()

    # 4. Target COMPILE_FLAGS (legacy)
    get_target_property(target_flags ${target} COMPILE_FLAGS)
    if(target_flags)
        extract_march_from_string("${target_flags}")
    endif()

    # 5. Target COMPILE_OPTIONS (highest priority)
    get_target_property(target_opts ${target} COMPILE_OPTIONS)
    if(target_opts)
        extract_march_from_list("${target_opts}")
    endif()

    set(${result_var} "${march}" PARENT_SCOPE)
endfunction()

# On arm, the architecture is passed with `-march`, including options.
# We need to add flags, such as `+sve` without breaking the base arch.
# So wee look everywhere in CMake for what it may be set too and extend it.
function(xsimd_arm_compile_args out_var arch baseline_arch)
    xsimd_normalize_arch(arch "${arch}")
    if("${arch}" MATCHES "NEON|NEON64")
        set(${out_var} "${baseline_arch}+neon" PARENT_SCOPE)
    elseif("${arch}" STREQUAL "SVE([0-9]+)")
        set(${out_var} "-march=${baseline_arch}+sve" "-msve-vector-bits=${CMAKE_MATCH_1}" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Unknown xsimd architecture: ${arch}")
    endif()
endfunction()


function(xsimd_target_compile_with target scope arch)
    # Validate scope
    if(NOT scope STREQUAL "PUBLIC" AND NOT scope STREQUAL "PRIVATE" AND NOT scope STREQUAL "INTERFACE")
        message(FATAL_ERROR "scope must be PUBLIC, PRIVATE, or INTERFACE, got: ${scope}")
    endif()

    # TODO factor this into a generic function independent of platform 
    if(
        CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|x86|i[3-6]86|AMD64|amd64" OR
        CMAKE_CXX_COMPILER_TARGET MATCHES "x86_64|x86|i[3-6]86|AMD64|amd64"
    )
        xsimd_x86_compile_args(compile_args "${arch}")
    elseif(
        CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64" OR
        CMAKE_CXX_COMPILER_TARGET MATCHES "aarch64|arm64|ARM64"
    )
        get_effective_march(${target} baseline_arch)
        if(NOT baseline_arch)
            set(effective_march "armv8-a")
        endif()
        xsimd_arm_compile_args(new_arch "${arch}" "${baseline_arch}")
    else()
    endif()

    target_compile_options(${target} ${scope} ${compile_args})
endfunction()


function(xsimd_files_compile_with file_or_dir arch)
    # Names of option parameters (without arguments)
    set(options)
    # Names of named parameters with a single argument
    set(one_value_args)
    # Names of named parameters with a multiple arguments
    set(multi_values_args)
    cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_values_args}" ${ARGN})
    # Extra arguments not accounted for
    if(ARG_UNPARSED_ARGUMENTS)
        message(
            AUTHOR_WARNING
            "Unrecoginzed options passed to ${CMAKE_CURRENT_FUNCTION}: "
            "${ARG_UNPARSED_ARGUMENTS}"
        )
    endif()

    xsimd_arch_compile_args(compile_args "${arch}")

    if(NOT EXISTS "${file_or_dir}")
        message(FATAL_ERROR "File or directory ${file_or_dir} does not exist")
    endif()

    if(IS_DIRECTORY "${file_or_dir}")
        set_source_files_properties(
            "${file_or_dir}" DIRECTORY "${file_or_dir}" PROPERTIES COMPILE_OPTIONS "${compile_args}"
        )
    else()
        set_source_files_properties(
            "${file_or_dir}" PROPERTIES COMPILE_OPTIONS "${compile_args}"
        )
    endif()

endfunction()

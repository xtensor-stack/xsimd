include(CheckCXXCompilerFlag)


function(xsimd_harden_trivial_auto_var_init target scope)
    # Names of option parameters (without arguments)
    set(options)
    # Names of named parameters with a single argument
    set(one_value_args PATTERN)
    # Names of named parameters with a multiple arguments
    set(multi_values_args)
    cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_values_args}" ${ARGN})
    if(ARG_UNPARSED_ARGUMENTS)
        message(
            AUTHOR_WARNING
            "Unrecoginzed options passed to ${CMAKE_CURRENT_FUNCTION}: "
            "${ARG_UNPARSED_ARGUMENTS}"
        )
    endif()

    if(NOT scope STREQUAL "PUBLIC" AND NOT scope STREQUAL "PRIVATE" AND NOT scope STREQUAL "INTERFACE")
        message(FATAL_ERROR "scope must be PUBLIC, PRIVATE, or INTERFACE, got: ${scope}")
    endif()

    if(NOT XSIMD_HARDEN_TRIVIAL_AUTO_VAR_INIT)
        return()
    endif()

    if(NOT ARG_PATTERN)
        set(ARG_PATTERN "pattern")
    endif()

    set(flag "-ftrivial-auto-var-init=${ARG_PATTERN}")
    check_cxx_compiler_flag("${flag}" XSIMD_HAS_FTRIVIAL_AUTO_VAR_INIT_${ARG_PATTERN})
    if(XSIMD_HAS_FTRIVIAL_AUTO_VAR_INIT_${ARG_PATTERN})
        target_compile_options(${target} ${scope} "${flag}")
    endif()
endfunction()

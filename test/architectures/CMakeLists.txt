set(INTEL_PROCESSORS
    knl knm skylake-avx512 cannonlake icelake-client
    icelake-server cascadelake cooperlake tigerlake sapphirerapids alderlake
    rocketlake graniterapids graniterapids-d znver4)
set(CMAKE_CXX_FLAGS "")

foreach(INTEL_PROCESSOR ${INTEL_PROCESSORS})
    check_cxx_compiler_flag(-march=${INTEL_PROCESSOR} FLAG_SUPPORTED_${INTEL_PROCESSOR})
    if(FLAG_SUPPORTED_${INTEL_PROCESSOR})
        message(STATUS ${INTEL_PROCESSOR})
        add_library(test_${INTEL_PROCESSOR} OBJECT dummy.cpp)
        target_compile_options(test_${INTEL_PROCESSOR} PRIVATE -march=${INTEL_PROCESSOR})
        target_include_directories(test_${INTEL_PROCESSOR} PRIVATE ${XSIMD_INCLUDE_DIR})
        add_dependencies(xtest test_${INTEL_PROCESSOR})
        if(ENABLE_XTL_COMPLEX)
            target_compile_features(test_${INTEL_PROCESSOR} PRIVATE cxx_std_14)
            target_compile_definitions(test_${INTEL_PROCESSOR} PRIVATE XSIMD_ENABLE_XTL_COMPLEX=1)
            target_link_libraries(test_${INTEL_PROCESSOR} PRIVATE xtl)
        endif()
    endif()
endforeach()


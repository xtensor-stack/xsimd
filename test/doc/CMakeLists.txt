# Only test under some architecture, because it's just a sanity check, no full
# coverage is needed.

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64" AND NOT CMAKE_OSX_ARCHITECTURES)

if(ENABLE_XTL_COMPLEX)
  add_compile_definitions(XSIMD_ENABLE_XTL_COMPLEX=1)
  include_directories(${xtl_INCLUDE_DIRS})
endif()

add_library(test_doc_any_arch OBJECT
            explicit_use_of_an_instruction_set_mean_aligned.cpp
            explicit_use_of_an_instruction_set_mean_arch_independent.cpp
            explicit_use_of_an_instruction_set_mean.cpp
            explicit_use_of_an_instruction_set_mean_tag_dispatch.cpp
            manipulating_abstract_batches.cpp
            manipulating_parametric_batches.cpp
            writing_vectorized_code.cpp)
target_include_directories(test_doc_any_arch PRIVATE ${XSIMD_INCLUDE_DIR})
target_compile_options(test_doc_any_arch PRIVATE -mavx)

add_library(test_doc_avx2 OBJECT
            explicit_use_of_an_instruction_set.cpp
            sum_avx2.cpp)
target_compile_options(test_doc_avx2 PRIVATE -mavx2)
target_include_directories(test_doc_avx2 PRIVATE ${XSIMD_INCLUDE_DIR})

add_library(test_doc_sse2 OBJECT
            sum_sse2.cpp)
target_compile_options(test_doc_sse2 PRIVATE -msse2)
target_include_directories(test_doc_sse2 PRIVATE ${XSIMD_INCLUDE_DIR})

add_dependencies(xtest test_doc_any_arch test_doc_avx2 test_doc_sse2)

endif()

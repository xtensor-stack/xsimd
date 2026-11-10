cmake_minimum_required(VERSION 3.19)
project(fetchcontent-test VERSION 0.0.1)

include(FetchContent)
FetchContent_Declare(xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG 14.3.0)
FetchContent_MakeAvailable(xsimd)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE xsimd)

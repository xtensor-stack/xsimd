cmake_minimum_required(VERSION 3.19)
project(test VERSION 0.0.1)

find_package(xsimd REQUIRED)

add_executable(test main.cpp)
target_link_libraries(test PUBLIC xsimd)

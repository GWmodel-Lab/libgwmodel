cmake_minimum_required(VERSION 3.1)
project(gwmodel)

set(CMAKE_CXX_STARDARD 11)

option(BUILD_CXX_LIBRARY "Weather to build a C++ library instead of a C library" OFF)

add_subdirectory(${CMAKE_SOURCE_DIR}/src)

enable_testing()

add_subdirectory(${CMAKE_SOURCE_DIR}/test)
add_test(
    NAME testBasicGWR 
    COMMAND $<TARGET_FILE:testBasicGWR> --success
)
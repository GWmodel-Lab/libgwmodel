cmake_minimum_required(VERSION 3.1)
project(gwmodel)

set(CMAKE_CXX_STARDARD 11)

option(BUILD_SHARED_LIBRARY "Weather to build a shared library" OFF) 
if(BUILD_SHARED_LIBRARY)
    message(WARNING "The library can only be used by C-style interfaces.")
endif(BUILD_SHARED_LIBRARY)


add_subdirectory(${CMAKE_SOURCE_DIR}/src)

enable_testing()

add_subdirectory(${CMAKE_SOURCE_DIR}/test)
add_test(
    NAME testBasicGWR 
    COMMAND $<TARGET_FILE:testBasicGWR> --success
)
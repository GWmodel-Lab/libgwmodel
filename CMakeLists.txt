cmake_minimum_required(VERSION 3.12)
project(libgwmodel)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(Armadillo REQUIRED)
if(ARMADILLO_FOUND)
    add_definitions(-DARMA_USE_LAPACK -DARMA_USE_BLAS -DARMA_DONT_USE_WRAPPER)
    include_directories(${ARMADILLO_INCLUDE_DIRS})

    find_package(BLAS)
    if(NOT BLAS_FOUND)
        message(SEND_ERROR "BLAS Not Found!")
    endif(NOT BLAS_FOUND)
    
    find_package(LAPACK)
    if(NOT LAPACK_FOUND)
        message(SEND_ERROR "LAPACK Not Found!")
    endif(NOT LAPACK_FOUND)
else(ARMADILLO_FOUND)
    message(FATAL_ERROR "Armadillo Not Found!")
endif(ARMADILLO_FOUND)

option(BUILD_SHARED_LIBS "Weather to build a shared library" OFF) 

enable_testing()

if(BUILD_SHARED_LIBS)
    message(STATUS "The library can only be used by C-style interfaces.")
endif(BUILD_SHARED_LIBS)


if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(MSVC)

set(LIBGWMODEL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(${CMAKE_SOURCE_DIR}/src)
add_subdirectory(${CMAKE_SOURCE_DIR}/test)

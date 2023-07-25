cmake_minimum_required(VERSION 3.12)
project(libgwmodel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH cmake)

SET (ENABLE_OpenMP TRUE CACHE BOOL "Determines whether OpemMP support should be built")
SET (WITH_TESTS TRUE CACHE BOOL "Determines whether to build and run tests")

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

if(MSVC)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(MSVC)

set(LIBGWMODEL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(src)

if(WITH_TESTS)
enable_testing()
add_subdirectory(test)
endif()

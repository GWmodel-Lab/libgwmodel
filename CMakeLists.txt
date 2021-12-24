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
option(WITH_PYTHON "Whether to build python extension" OFF)
option(USE_SKBUILD "Whether to use scikit-build as building and packaging tools" OFF)

if(USE_SKBUILD)
    set(WITH_PYTHON ON)
endif()


# if(WITH_PYTHON)
#     # set(BUILD_SHARED_LIBS ON)
# endif(WITH_PYTHON)

if(BUILD_SHARED_LIBS)
    message(STATUS "The library can only be used by C-style interfaces.")
endif(BUILD_SHARED_LIBS)


if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(MSVC)

set(LIBGWMODEL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(${CMAKE_SOURCE_DIR}/src)

if(WITH_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
    message(STATUS "Python Interpreter: ${Python3_EXECUTABLE}")
    add_subdirectory(${CMAKE_SOURCE_DIR}/python)
else()
    set(USE_SKBUILD OFF)
endif()


enable_testing()
add_subdirectory(${CMAKE_SOURCE_DIR}/test)

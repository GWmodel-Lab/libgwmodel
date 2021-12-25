cmake_minimum_required(VERSION 3.12)
project(pygwmodel)

if(SKBUILD)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
endif(SKBUILD)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

if(SKBUILD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
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
    include_directories(${LIBGWMODEL_INCLUDE_DIR}/gwmodelpp)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src ${pygwmodel_BINARY_DIR}/libgwmodel)
    add_subdirectory("pygwmodel")
    enable_testing()
    add_subdirectory("test")
elseif(USE_SKBUILD)
    set(PYGWMODEL_SKBUILD_DIR ${CMAKE_BINARY_DIR}/pygwmodel)
    add_custom_target(pygwmodel_skbuild
        VERBATIM
        COMMAND_EXPAND_LISTS
        COMMAND ${CMAKE_COMMAND} -E
            make_directory
            ${PYGWMODEL_SKBUILD_DIR}
        COMMAND ${CMAKE_COMMAND} -E
            copy_directory ${CMAKE_SOURCE_DIR}/python ${PYGWMODEL_SKBUILD_DIR}
        COMMAND ${CMAKE_COMMAND} -E
            copy_directory ${CMAKE_SOURCE_DIR}/cmake ${PYGWMODEL_SKBUILD_DIR}/cmake
        COMMAND ${CMAKE_COMMAND} -E
            copy_directory ${CMAKE_SOURCE_DIR}/include ${PYGWMODEL_SKBUILD_DIR}/include
        COMMAND ${CMAKE_COMMAND} -E
            copy_directory ${CMAKE_SOURCE_DIR}/src ${PYGWMODEL_SKBUILD_DIR}/src
    )
else()
    include_directories(${LIBGWMODEL_INCLUDE_DIR}/gwmodelpp)
    add_subdirectory("pygwmodel")
    add_subdirectory("test")
endif()

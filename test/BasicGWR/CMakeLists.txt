cmake_minimum_required(VERSION 3.1)

project(testBasicGWR LANGUAGES CXX)

find_package(Catch2 REQUIRED)

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

add_executable(testBasicGWR main.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testBasicGWR PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2)
cmake_minimum_required(VERSION 3.1)
project(libgwmodel)

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

find_package(OpenMP)
if(OpenMP_FOUND AND OpenMP_C_FOUND AND OpenMP_CXX_FOUND)
    SET (ENABLE_OpenMP TRUE CACHE BOOL "Determines whether OpemMP support should be built")
    if(ENABLE_OpenMP)
        add_definitions(-DENABLE_OPENMP)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS"${CMAKE_EXE_LINKER_FLAGS}${OpenMP_EXE_LINKER_FLAGS}")
        include_directories(${OpenMP_C_INCLUDE_DIRS} ${OpenMP_CXX_INCLUDE_DIRS})
    endif(ENABLE_OpenMP)
endif(OpenMP_FOUND AND OpenMP_C_FOUND AND OpenMP_CXX_FOUND)

include_directories(
    ${CMAKE_SOURCE_DIR}/include 
    ${CMAKE_SOURCE_DIR}/include/gwmodelpp
)

add_definitions(-DM_PI=3.14159265358979323846 -DM_PI_2=1.57079632679489661923 -DDOUBLE_EPS=1e-8)

set(SOURCES_CXX
    gwmodelpp/spatialweight/CGwmBandwidthWeight.cpp
    gwmodelpp/spatialweight/CGwmCRSDistance.cpp
    gwmodelpp/spatialweight/CGwmDistance.cpp
    gwmodelpp/spatialweight/CGwmDMatDistance.cpp
    gwmodelpp/spatialweight/CGwmMinkwoskiDistance.cpp
    gwmodelpp/spatialweight/CGwmSpatialWeight.cpp
    gwmodelpp/spatialweight/CGwmWeight.cpp

    gwmodelpp/CGwmAlgorithm.cpp
    gwmodelpp/CGwmBandwidthSelector.cpp
    gwmodelpp/CGwmVariableForwardSelector.cpp
    gwmodelpp/CGwmSimpleLayer.cpp
    gwmodelpp/CGwmSpatialAlgorithm.cpp
    gwmodelpp/CGwmSpatialMonoscaleAlgorithm.cpp
    gwmodelpp/CGwmGWRBase.cpp
    gwmodelpp/CGwmGWRBasic.cpp
)

set(SOURCES_C
    gwmodel.cpp
)

add_library(gwmodel SHARED ${SOURCES_CXX} ${SOURCES_C})

target_link_libraries(gwmodel
    ${ARMADILLO_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

if(OpenMP_FOUND)
    target_link_libraries(gwmodel
        PRIVATE
        ${OpenMP_C_LIBRARIES}
        ${OpenMP_CXX_LIBRARIES}
    )
endif(OpenMP_FOUND)
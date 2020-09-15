cmake_minimum_required(VERSION 3.1)
project(libgwmodel)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

include_directories(
    ${CMAKE_SOURCE_DIR}/include 
    ${CMAKE_SOURCE_DIR}/include/gwmodelpp
)

add_definitions(-DM_PI=3.14159265358979323846 -DM_PI_2=1.57079632679489661923 -DDOUBLE_EPS=1e-8)

set(SOURCES
gwmodelpp/spatialweight/CGwmBandwidthWeight.cpp
gwmodelpp/spatialweight/CGwmCRSDistance.cpp
gwmodelpp/spatialweight/CGwmDistance.cpp
gwmodelpp/spatialweight/CGwmDMatDistance.cpp
gwmodelpp/spatialweight/CGwmMinkwoskiDistance.cpp
gwmodelpp/spatialweight/CGwmSpatialWeight.cpp
gwmodelpp/spatialweight/CGwmWeight.cpp

gwmodelpp/CGwmSimpleLayer.cpp
gwmodelpp/CGwmSpatialAlgorithm.cpp
gwmodelpp/CGwmSpatialMonoscaleAlgorithm.cpp
gwmodelpp/CGwmGWRBase.cpp
gwmodelpp/CGwmGWRBasic.cpp

gwmodel.cpp
)

add_library(gwmodel SHARED ${SOURCES})
target_link_libraries(gwmodel ${ARMADILLO_LIBRARIES})
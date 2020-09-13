cmake_minimum_required(VERSION 3.1)
project(libgwmodel)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

include_directories(
    ${CMAKE_SOURCE_DIR}/include 
    ${CMAKE_SOURCE_DIR}/include/gwmodelpp
)

set(SOURCES
gwmodelpp/CSimpleLayer.cpp
gwmodelpp/CSpatialAlgorithm.cpp
gwmodel.cpp
)

add_library(gwmodel SHARED ${SOURCES})
target_link_libraries(gwmodel ${ARMADILLO_LIBRARIES})
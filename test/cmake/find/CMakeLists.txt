cmake_minimum_required(VERSION 3.12)
project(TestFindGWmodel)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(GWmodel REQUIRED)

if(GWmodel_FOUND)
    message(STATUS "GWmodel library is found!")
    message(STATUS "Include directory: ${GWmodel_INCLUDE_DIRS}")
    message(STATUS "Library: ${GWmodel_LIBRARIES}")
else()
    message(FATAL_ERROR "Cannot find GWmodel library!")
endif()

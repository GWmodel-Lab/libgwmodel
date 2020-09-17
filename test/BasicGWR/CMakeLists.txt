cmake_minimum_required(VERSION 3.1)

project(testBasicGWR)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include/gwmodelpp)

add_executable(testBasicGWR main.cpp)
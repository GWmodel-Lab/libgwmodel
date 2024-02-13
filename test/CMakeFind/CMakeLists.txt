cmake_minimum_required(VERSION 3.12)
project(test_cmake_find)

set(CMAKE_CXX_STANDARD 17)

find_package(libgwmodel REQUIRED)
include_directories(${LIBGWMODEL_INCLUDE_DIRS})

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_executable(main main.cpp)
target_link_libraries(main PRIVATE gwmodel ${ARMADILLO_LIBRARIES})

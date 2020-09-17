cmake_minimum_required(VERSION 3.1)

project(testBasicGWR LANGUAGES CXX)

find_package(Catch2 REQUIRED)

add_executable(testBasicGWR main.cpp)
target_link_libraries(testBasicGWR gwmodel Catch2::Catch2)
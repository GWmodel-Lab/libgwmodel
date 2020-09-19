cmake_minimum_required(VERSION 3.1)

project(testBasicGWR LANGUAGES CXX)

add_executable(testBasicGWR main.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testBasicGWR PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2)
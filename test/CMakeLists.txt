cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/gwmodelpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${gwmodel_BINARY_DIR}/test)

add_subdirectory(${CMAKE_SOURCE_DIR}/test/BasicGWR)

set(DATA_LONDONHP100
    ${CMAKE_CURRENT_SOURCE_DIR}/data/londonhp100coords.csv
    ${CMAKE_CURRENT_SOURCE_DIR}/data/londonhp100data.csv
)

file(COPY ${DATA_LONDONHP100} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data)
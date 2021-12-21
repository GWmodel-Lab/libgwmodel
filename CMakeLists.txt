cmake_minimum_required(VERSION 3.12)
project(gwmodel)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

option(BUILD_SHARED_LIBS "Weather to build a shared library" OFF) 
option(WITH_PYTHON "Whether to build python extension" OFF)

if(WITH_PYTHON)
    set(BUILD_SHARED_LIBS ON)
endif(WITH_PYTHON)

if(BUILD_SHARED_LIBS)
    message(STATUS "The library can only be used by C-style interfaces.")
endif(BUILD_SHARED_LIBS)


if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(MSVC)

add_subdirectory(${CMAKE_SOURCE_DIR}/src)

if(WITH_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    message(STATUS "Python Interpreter: ${Python3_EXECUTABLE}")
    add_subdirectory(${CMAKE_SOURCE_DIR}/pygwmodel)
endif(WITH_PYTHON)

enable_testing()
add_subdirectory(${CMAKE_SOURCE_DIR}/test)


install(TARGETS gwmodel
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

if(WITH_PYTHON)
    install(TARGETS mat_interface name_list_interface simple_layer spatial_weight variable_interface gwr_basic gwss gwpca
            LIBRARY DESTINATION pygwmodel)
    install(FILES ${CMAKE_SOURCE_DIR}/pygwmodel/__init__.py DESTINATION pygwmodel)
endif(WITH_PYTHON)

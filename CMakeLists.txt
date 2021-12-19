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
    find_package(Python3 COMPONENTS Interpreter Development)
    add_subdirectory(${CMAKE_SOURCE_DIR}/pygwmodel)
endif(WITH_PYTHON)

enable_testing()

add_subdirectory(${CMAKE_SOURCE_DIR}/test)
add_test(
    NAME testBasicGWR 
    COMMAND $<TARGET_FILE:testBasicGWR> --success
)
add_test(
    NAME testGWSS 
    COMMAND $<TARGET_FILE:testGWSS> --success
)
add_test(
    NAME testGWPCA 
    COMMAND $<TARGET_FILE:testGWPCA> --success
)
add_test(
    NAME testMultithread 
    COMMAND $<TARGET_FILE:testMultithread> --success
)
if(WITH_PYTHON)
    set(PYTHON_TEST_DATA ${CMAKE_SOURCE_DIR}/test/data/londonhp100.csv)
    if(WIN32)
        set(PYTHON_TEST_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/build/bin/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>)
    else()
        set(PYTHON_TEST_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/build/bin)
    endif()
    message(STATUS "Python Interpreter: ${Python3_EXECUTABLE}")
    add_test(
        NAME testPythonGWRBasic
        COMMAND ${Python3_EXECUTABLE} testGWRBasic.py ${PYTHON_TEST_DATA}
        WORKING_DIRECTORY ${PYTHON_TEST_SCRIPT_DIR}
    )
    add_test(
        NAME testPythonGWSS
        COMMAND ${Python3_EXECUTABLE} testGWSS.py ${PYTHON_TEST_DATA}
        WORKING_DIRECTORY ${PYTHON_TEST_SCRIPT_DIR}
    )
    add_test(
        NAME testPythonGWPCA
        COMMAND ${Python3_EXECUTABLE} testGWPCA.py ${PYTHON_TEST_DATA}
        WORKING_DIRECTORY ${PYTHON_TEST_SCRIPT_DIR}
    )
endif(WITH_PYTHON)


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

cmake_minimum_required(VERSION 3.1)
project(testPython LANGUAGES CXX)

if(WIN32)
    set(PYTHON_TEST_SCRIPT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>)
else()
    set(PYTHON_TEST_SCRIPT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

add_custom_target(testPythonDeps ALL 
    VERBATIM 
    COMMAND_EXPAND_LISTS
    COMMAND ${CMAKE_COMMAND} -E
        make_directory
        ${PYTHON_TEST_SCRIPT_DIR}/pygwmodel
    COMMAND ${CMAKE_COMMAND} -E
        copy_if_different
        ${CMAKE_SOURCE_DIR}/pygwmodel/__init__.py
        ${PYTHON_TEST_SCRIPT_DIR}/pygwmodel
    COMMAND ${CMAKE_COMMAND} -E
        copy_if_different
        $<TARGET_FILE:pygwmodel>
        ${PYTHON_TEST_SCRIPT_DIR}/pygwmodel
    COMMAND ${CMAKE_COMMAND} -E
        copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/testGWRBasic.py
        ${CMAKE_CURRENT_SOURCE_DIR}/testGWSS.py
        ${CMAKE_CURRENT_SOURCE_DIR}/testGWPCA.py
        ${PYTHON_TEST_SCRIPT_DIR}
)

set(PYTHON_TEST_DATA ${CMAKE_SOURCE_DIR}/test/data/londonhp100.csv)
# add_test(
#     NAME testPythonGWRBasic
#     COMMAND ${Python3_EXECUTABLE} testGWRBasic.py ${PYTHON_TEST_DATA}
#     WORKING_DIRECTORY ${PYTHON_TEST_SCRIPT_DIR}
# )
# add_test(
#     NAME testPythonGWSS
#     COMMAND ${Python3_EXECUTABLE} testGWSS.py ${PYTHON_TEST_DATA}
#     WORKING_DIRECTORY ${PYTHON_TEST_SCRIPT_DIR}
# )
add_test(
    NAME testPythonGWPCA
    COMMAND ${Python3_EXECUTABLE} testGWPCA.py ${PYTHON_TEST_DATA}
    WORKING_DIRECTORY ${PYTHON_TEST_SCRIPT_DIR}
)
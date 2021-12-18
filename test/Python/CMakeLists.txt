cmake_minimum_required(VERSION 3.1)
project(testPython LANGUAGES CXX)

if(WIN32)
    set(PYGWMODEL_COPY_DIR ${PYGWMODEL_COPY_DIR})
else()
    set(PYGWMODEL_COPY_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

add_custom_target(testPythonDeps ALL 
    VERBATIM 
    COMMAND_EXPAND_LISTS
    COMMAND ${CMAKE_COMMAND} -E
        make_directory
        ${PYGWMODEL_COPY_DIR}/pygwmodel
    COMMAND ${CMAKE_COMMAND} -E
        copy_if_different
        ${CMAKE_SOURCE_DIR}/python/__init__.py
        ${PYGWMODEL_COPY_DIR}/pygwmodel
    COMMAND ${CMAKE_COMMAND} -E
        copy_if_different
        $<TARGET_FILE:gwss>
        $<TARGET_FILE:mat_interface>
        $<TARGET_FILE:name_list_interface>
        $<TARGET_FILE:simple_layer>
        $<TARGET_FILE:spatial_weight>
        $<TARGET_FILE:variable_interface>
        $<TARGET_FILE:gwr_basic>
        ${PYGWMODEL_COPY_DIR}/pygwmodel
    COMMAND ${CMAKE_COMMAND} -E
        copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/testGWSS.py
        ${PYGWMODEL_COPY_DIR}
)

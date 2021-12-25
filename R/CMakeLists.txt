cmake_minimum_required(VERSION 3.12)
project(gwmodelr)

set(PROJECT_VERSION_R "0.2-0")


set(GWMODELR_RBUILD_DIR ${CMAKE_BINARY_DIR}/gwmodelr)
make_directory(${GWMODELR_RBUILD_DIR})

configure_file(DESCRIPTION.in ${GWMODELR_RBUILD_DIR}/DESCRIPTION)
configure_file(NAMESPACE.in ${GWMODELR_RBUILD_DIR}/NAMESPACE)

add_custom_target(gwmodelr_rbuild
    VERBATIM
    COMMAND_EXPAND_LISTS
    COMMAND ${CMAKE_COMMAND} -E
        make_directory
        ${GWMODELR_RBUILD_DIR}
    COMMAND ${CMAKE_COMMAND} -E
        copy_directory ${CMAKE_SOURCE_DIR}/R/scripts ${GWMODELR_RBUILD_DIR}/R
    COMMAND ${CMAKE_COMMAND} -E
        copy_directory ${CMAKE_SOURCE_DIR}/R/src ${GWMODELR_RBUILD_DIR}/src
    COMMAND ${CMAKE_COMMAND} -E
        copy_directory ${CMAKE_SOURCE_DIR}/include ${GWMODELR_RBUILD_DIR}/src/libgwmodel
    COMMAND ${CMAKE_COMMAND} -E
        copy_directory ${CMAKE_SOURCE_DIR}/src ${GWMODELR_RBUILD_DIR}/src/libgwmodel
    WORKING_DIRECTORY ${GWMODELR_RBUILD_DIR}
)

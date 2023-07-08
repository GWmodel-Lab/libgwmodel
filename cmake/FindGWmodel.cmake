# FindGWmodel
# --------
#
# Find the GWmodel libraries
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# The following variables will be defined:
#
# ``GWmodel_FOUND`` True if GWmodel found on the local system
#
# ``GWmodel_INCLUDE_DIR`` Location of GWmodel header files
#
# ``GWmodel_LIBRARY_DIR`` Location of GWmodel libraries
#
# ``GWmodel_LIBRARIES`` List of the GWmodel libraries found
#

if(DEFINED ENV{USR} AND NOT "$ENV{USR}" EQUAL "")
    set(USR_DIR "$ENV{USR}")
else()
    set(USR_DIR "/usr")
endif()

if(DEFINED ENV{USR_LOCAL} AND NOT "$ENV{USR_LOCAL}" EQUAL "")
    set(USR_LOCAL_DIR "$ENV{USR_LOCAL}")
else()
    set(USR_LOCAL_DIR "/usr/local")
endif()


set(GWmodel_SEARCH_PATHS
    "${USR_DIR}"
    "${USR_LOCAL_DIR}"
    "${CMAKE_PREFIX_PATH}"
    "$ENV{LIB_DIR}"
    "$ENV{LIB}"
)

find_path(GWmodel_INCLUDE_DIRS
    NAMES gwmodel.h
    PATHS ${GWmodel_SEARCH_PATHS}
    PATH_SUFFIXES "include"
)


find_library(GWmodel_LIBRARIES
    NAMES gwmodel.lib libgwmodel.a libgwmodel.lib
    PATHS ${GWmodel_SEARCH_PATHS}
    PATH_SUFFIXES "lib"
)

if(GWmodel_INCLUDE_DIRS AND GWmodel_LIBRARIES)
    set(GWmodel_FOUND TRUE CACHE BOOL "Whether GWmodel is Found")
endif()

if(GWmodel_FOUND)
    message(STATUS "Found GWmodel: ${GWmodel_LIBRARIES}")
else()
    if(GWmodel_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find GWmodel")
    endif(GWmodel_FIND_REQUIRED)
endif()

cmake_minimum_required(VERSION 3.12)

find_package(OpenMP)
if(OpenMP_FOUND AND OpenMP_C_FOUND AND OpenMP_CXX_FOUND)
    if(ENABLE_OpenMP)
        add_definitions(-DENABLE_OPENMP)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS"${CMAKE_EXE_LINKER_FLAGS}${OpenMP_EXE_LINKER_FLAGS}")
        include_directories(${OpenMP_C_INCLUDE_DIRS} ${OpenMP_CXX_INCLUDE_DIRS})
    endif(ENABLE_OpenMP)
endif(OpenMP_FOUND AND OpenMP_C_FOUND AND OpenMP_CXX_FOUND)

find_package(GSL REQUIRED)
if(GSL_FOUND)
    include_directories(${GSL_INCLUDE_DIRS})
else()
    message(FATAL "GSL not found!")
endif()


include_directories(
    ${LIBGWMODEL_INCLUDE_DIR} 
    ${LIBGWMODEL_INCLUDE_DIR}/gwmodelpp
)

add_definitions(-DM_PI=3.14159265358979323846 -DM_PI_2=1.57079632679489661923 -DDOUBLE_EPS=1e-8)
if(NOT MSVC)
    add_definitions(-DDBL_MAX=__DBL_MAX__)    
endif(NOT MSVC)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-D_DEBUG)
endif()


set(SOURCES_CXX
    gwmodelpp/spatialweight/CGwmBandwidthWeight.cpp
    gwmodelpp/spatialweight/CGwmCRSDistance.cpp
    gwmodelpp/spatialweight/CGwmOneDimDistance.cpp
    gwmodelpp/spatialweight/CGwmDistance.cpp
    gwmodelpp/spatialweight/CGwmDMatDistance.cpp
    gwmodelpp/spatialweight/CGwmMinkwoskiDistance.cpp
    gwmodelpp/spatialweight/CGwmSpatialWeight.cpp
    gwmodelpp/spatialweight/CGwmWeight.cpp

    gwmodelpp/CGwmBandwidthSelector.cpp
    gwmodelpp/CGwmVariableForwardSelector.cpp
    gwmodelpp/CGwmSpatialAlgorithm.cpp
    gwmodelpp/CGwmSpatialMonoscaleAlgorithm.cpp
    gwmodelpp/CGwmSpatialMultiscaleAlgorithm.cpp
    gwmodelpp/CGwmGWRBase.cpp
    gwmodelpp/CGwmGWRBasic.cpp
    gwmodelpp/CGwmGWSS.cpp
    gwmodelpp/CGwmGWPCA.cpp
    gwmodelpp/CGwmGWDR.cpp
    gwmodelpp/CGwmMGWR.cpp
    gwmodelpp/GwmLogger.cpp
    gwmodelpp/CGwmRobustGWR.cpp
)

set(SOURCES_C
    gwmodel.cpp
)

set(HEADERS_CXX
    ../include/gwmodelpp/spatialweight/CGwmBandwidthWeight.h
    ../include/gwmodelpp/spatialweight/CGwmCRSDistance.h
    ../include/gwmodelpp/spatialweight/CGwmOneDimDistance.h
    ../include/gwmodelpp/spatialweight/CGwmDistance.h
    ../include/gwmodelpp/spatialweight/CGwmDMatDistance.h
    ../include/gwmodelpp/spatialweight/CGwmMinkwoskiDistance.h
    ../include/gwmodelpp/spatialweight/CGwmSpatialWeight.h
    ../include/gwmodelpp/spatialweight/CGwmWeight.h

    ../include/gwmodelpp/CGwmAlgorithm.h
    ../include/gwmodelpp/CGwmBandwidthSelector.h
    ../include/gwmodelpp/CGwmVariableForwardSelector.h
    ../include/gwmodelpp/CGwmSpatialAlgorithm.h
    ../include/gwmodelpp/CGwmSpatialMonoscaleAlgorithm.h
    ../include/gwmodelpp/CGwmSpatialMultiscaleAlgorithm.h
    ../include/gwmodelpp/CGwmGWRBase.h
    ../include/gwmodelpp/CGwmGWRBasic.h
    ../include/gwmodelpp/CGwmGWSS.h
    ../include/gwmodelpp/CGwmGWPCA.h
    ../include/gwmodelpp/CGwmGWDR.h
    ../include/gwmodelpp/CGwmMGWR.h
    ../include/gwmodelpp/GwmLogger.h
    ../include/gwmodelpp/CGwmRobustGWR.h
)

set(HEADERS_C
    ../include/gwmodel.h
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${gwmodel_BINARY_DIR}/bin)

add_library(gwmodel STATIC ${SOURCES_CXX} ${HEADERS_CXX} ${HEADERS_C})
set_property(TARGET gwmodel PROPERTY POSITION_INDEPENDENT_CODE ON)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    target_compile_options(gwmodel PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
        -Wall -Werror>
    $<$<CXX_COMPILER_ID:MSVC>:
        /W4 /wd4819 /WX>)
else()
    target_compile_options(gwmodel PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
        -Wall>
    $<$<CXX_COMPILER_ID:MSVC>:
        /W4 /wd4819>)
endif()

target_link_libraries(gwmodel
    ${ARMADILLO_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    GSL::gsl
    GSL::gslcblas
)

if(OpenMP_FOUND)
    target_link_libraries(gwmodel
        ${OpenMP_C_LIBRARIES}
        ${OpenMP_CXX_LIBRARIES}
    )
endif(OpenMP_FOUND)

install(TARGETS gwmodel
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY ../include/ DESTINATION include)
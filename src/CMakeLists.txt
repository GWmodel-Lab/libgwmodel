cmake_minimum_required(VERSION 3.12)

if(ENABLE_OpenMP)
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
endif()

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
    gwmodelpp/spatialweight/BandwidthWeight.cpp
    gwmodelpp/spatialweight/CRSDistance.cpp
    gwmodelpp/spatialweight/OneDimDistance.cpp
    gwmodelpp/spatialweight/Distance.cpp
    gwmodelpp/spatialweight/DMatDistance.cpp
    gwmodelpp/spatialweight/MinkwoskiDistance.cpp
    gwmodelpp/spatialweight/SpatialWeight.cpp
    gwmodelpp/spatialweight/Weight.cpp
    gwmodelpp/spatialweight/CRSSTDistance.cpp

    gwmodelpp/BandwidthSelector.cpp
    gwmodelpp/VariableForwardSelector.cpp
    gwmodelpp/SpatialAlgorithm.cpp
    gwmodelpp/SpatialMonoscaleAlgorithm.cpp
    gwmodelpp/SpatialMultiscaleAlgorithm.cpp
    gwmodelpp/GWRBase.cpp
    gwmodelpp/GWRBasic.cpp
    gwmodelpp/GWSS.cpp
    gwmodelpp/GWPCA.cpp
    gwmodelpp/GWDR.cpp
    gwmodelpp/GWRMultiscale.cpp
    gwmodelpp/GWRRobust.cpp
    gwmodelpp/GWRGeneralized.cpp
    gwmodelpp/BinomialModel.cpp
    gwmodelpp/PoissonModel.cpp
    gwmodelpp/GeneralizedLinearModel.cpp
    gwmodelpp/GWRScalable.cpp
    gwmodelpp/GWRLocalCollinearity.cpp
    gwmodelpp/GTWR.cpp
)

set(SOURCES_C
    gwmodel.cpp
)

set(HEADERS_CXX
    ../include/gwmodelpp/spatialweight/BandwidthWeight.h
    ../include/gwmodelpp/spatialweight/CRSDistance.h
    ../include/gwmodelpp/spatialweight/OneDimDistance.h
    ../include/gwmodelpp/spatialweight/Distance.h
    ../include/gwmodelpp/spatialweight/DMatDistance.h
    ../include/gwmodelpp/spatialweight/MinkwoskiDistance.h
    ../include/gwmodelpp/spatialweight/SpatialWeight.h
    ../include/gwmodelpp/spatialweight/Weight.h
    ../include/gwmodelpp/spatialweight/CRSSTDistance.h

    ../include/gwmodelpp/Algorithm.h
    ../include/gwmodelpp/BandwidthSelector.h
    ../include/gwmodelpp/VariableForwardSelector.h
    ../include/gwmodelpp/SpatialAlgorithm.h
    ../include/gwmodelpp/SpatialMonoscaleAlgorithm.h
    ../include/gwmodelpp/SpatialMultiscaleAlgorithm.h
    ../include/gwmodelpp/GWRBase.h
    ../include/gwmodelpp/GWRBasic.h
    ../include/gwmodelpp/GWSS.h
    ../include/gwmodelpp/GWPCA.h
    ../include/gwmodelpp/GWDR.h
    ../include/gwmodelpp/GWRMultiscale.h
    ../include/gwmodelpp/Logger.h
    ../include/gwmodelpp/GWRRobust.h
    ../include/gwmodelpp/GWRGeneralized.h
    ../include/gwmodelpp/BinomialModel.h
    ../include/gwmodelpp/LinearModel.h
    ../include/gwmodelpp/PoissonModel.h
    ../include/gwmodelpp/GeneralizedLinearModel.h
    ../include/gwmodelpp/GWRScalable.h
    ../include/gwmodelpp/GWRLocalCollinearity.h
    ../include/gwmodelpp/GTWR.h
)

set(HEADERS_C
    ../include/gwmodel.h
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${gwmodel_BINARY_DIR}/bin)

add_library(gwmodel STATIC ${SOURCES_CXX} ${HEADERS_CXX} ${HEADERS_C})
set_property(TARGET gwmodel PROPERTY POSITION_INDEPENDENT_CODE ON)

message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    if(CMAKE_CXX_COMPILER_ID STREQUAL Clang OR CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
        target_compile_options(gwmodel PRIVATE -Wmost -Werror)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
        target_compile_options(gwmodel PRIVATE /W4 /wd4819 /WX)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
        target_compile_options(gwmodel PRIVATE -Wall -Werror)
    endif()
else()
    if(CMAKE_CXX_COMPILER_ID STREQUAL Clang OR CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
        target_compile_options(gwmodel PRIVATE -Wmost)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
        target_compile_options(gwmodel PRIVATE /W4 /wd4819)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
        target_compile_options(gwmodel PRIVATE -Wall)
    endif()
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
cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Catch2 3 REQUIRED)

if(ENABLE_OpenMP)
    find_package(OpenMP)
    if(OpenMP_FOUND AND OpenMP_C_FOUND AND OpenMP_CXX_FOUND)
        if(ENABLE_OpenMP)
            if(MSVC)
                set(OpenMP_C_FLAGS "${OpenMP_C_FLAGS} -openmp:llvm")
                set(OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS} -openmp:llvm")
            endif(MSVC)
            add_definitions(-DENABLE_OPENMP)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
            set(CMAKE_EXE_LINKER_FLAGS"${CMAKE_EXE_LINKER_FLAGS}${OpenMP_EXE_LINKER_FLAGS}")
            include_directories(${OpenMP_C_INCLUDE_DIRS} ${OpenMP_CXX_INCLUDE_DIRS})
        endif(ENABLE_OpenMP)
    endif(OpenMP_FOUND AND OpenMP_C_FOUND AND OpenMP_CXX_FOUND)
endif()


include_directories(
    ${ARMADILLO_INCLUDE_DIR}
    ${LIBGWMODEL_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(SAMPLE_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)
add_definitions(-DSAMPLE_DATA_DIR="${SAMPLE_DATA_DIR}")

add_library(londonhp100 data/londonhp100.cpp)
target_link_libraries(londonhp100 ${ARMADILLO_LIBRARIES})

add_library(londonhp data/londonhp.cpp)
target_link_libraries(londonhp ${ARMADILLO_LIBRARIES})

add_library(telegram TerminateCheckTelegram.cpp FileTelegram.cpp)
target_link_libraries(telegram Catch2::Catch2WithMain)

add_executable(testGWRBasic testGWRBasic.cpp)
target_link_libraries(testGWRBasic PRIVATE gwmodel londonhp100 telegram ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWRBasic 
    COMMAND $<TARGET_FILE:testGWRBasic> --success --skip-benchmarks
)

add_executable(testGWAverage testGWAverage.cpp)
target_link_libraries(testGWAverage PRIVATE gwmodel londonhp100 telegram ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWAverage 
    COMMAND $<TARGET_FILE:testGWAverage> --success
)

add_executable(testGWCorrelation testGWCorrelation.cpp)
target_link_libraries(testGWCorrelation PRIVATE gwmodel londonhp100 telegram ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWCorrelation 
    COMMAND $<TARGET_FILE:testGWCorrelation> --success
)

add_executable(testGWPCA testGWPCA.cpp)
target_link_libraries(testGWPCA PRIVATE gwmodel londonhp100 telegram ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWPCA 
    COMMAND $<TARGET_FILE:testGWPCA> --success
)

add_executable(testSDR testSDR.cpp)
target_link_libraries(testSDR PRIVATE gwmodel londonhp100 telegram ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testSDR 
    COMMAND $<TARGET_FILE:testSDR> --success
)

add_executable(testGWRMultiscale testGWRMultiscale.cpp)
target_link_libraries(testGWRMultiscale PRIVATE gwmodel londonhp100 telegram ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWRMultiscale 
    COMMAND $<TARGET_FILE:testGWRMultiscale> --success
)

add_executable(testLogger testLogger.cpp)
target_link_libraries(testLogger PRIVATE gwmodel londonhp100 telegram ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testLogger 
    COMMAND $<TARGET_FILE:testLogger> --success
)

add_executable(testGWRRobust testGWRRobust.cpp)
target_link_libraries(testGWRRobust PRIVATE gwmodel londonhp100 telegram ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWRRobust 
    COMMAND $<TARGET_FILE:testGWRRobust> --success
)

add_executable(testGWRGeneralized testGWRGeneralized.cpp)
target_link_libraries(testGWRGeneralized PRIVATE gwmodel londonhp londonhp100 telegram ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWRGeneralized 
    COMMAND $<TARGET_FILE:testGWRGeneralized> --success
)

add_executable(testGWRScalable testGWRScalable.cpp)
target_link_libraries(testGWRScalable PRIVATE gwmodel londonhp100 telegram ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWRScalable 
    COMMAND $<TARGET_FILE:testGWRScalable> --success
)

add_executable(testGWRLocalCollinearity testGWRLocalCollinearity.cpp)
target_link_libraries(testGWRLocalCollinearity PRIVATE gwmodel londonhp100 telegram ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWRLocalCollinearity 
    COMMAND $<TARGET_FILE:testGWRLocalCollinearity> --success
)

add_executable(testGTWR testGTWR.cpp)
target_link_libraries(testGTWR PRIVATE gwmodel londonhp100 telegram ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGTWR 
    COMMAND $<TARGET_FILE:testGTWR> --success
)

add_executable(testGWDA testGWDA.cpp)
target_link_libraries(testGWDA PRIVATE gwmodel londonhp ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWDA 
    COMMAND $<TARGET_FILE:testGWDA> --success
)

if (ENABLE_CUDA)
add_executable(testCuMat testCuMat.cpp)
target_link_libraries(testCuMat PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testCuMat
    COMMAND $<TARGET_FILE:testCuMat> --success --skip-benchmarks
)
endif()

if (ENABLE_MPI)
add_subdirectory("./mpi")
endif()

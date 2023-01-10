cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Catch2 3 REQUIRED)

if(ENABLE_OpenMP)
    add_definitions(-DENABLE_OPENMP)
endif()


include_directories(
    ${ARMADILLO_INCLUDE_DIR}
    ${LIBGWMODEL_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(SAMPLE_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)
add_definitions(-DSAMPLE_DATA_DIR="${SAMPLE_DATA_DIR}")
set(SAMPL_LONDONHP100 ${CMAKE_CURRENT_SOURCE_DIR}/data/londonhp100.cpp)

add_definitions(-DSAMPLE_DATA_DIR="${SAMPLE_DATA_DIR}")
set(SAMPL_LONDONHP ${CMAKE_CURRENT_SOURCE_DIR}/data/londonhp.cpp)

add_executable(testBasicGWR BasicGWR.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testBasicGWR PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testBasicGWR 
    COMMAND $<TARGET_FILE:testBasicGWR> --success
)

add_executable(testGWSS GWSS.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testGWSS PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWSS 
    COMMAND $<TARGET_FILE:testGWSS> --success
)

add_executable(testGWPCA GWPCA.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testGWPCA PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWPCA 
    COMMAND $<TARGET_FILE:testGWPCA> --success
)

add_executable(testGWDR GWDR.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testGWDR PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWDR 
    COMMAND $<TARGET_FILE:testGWDR> --success
)

add_executable(testGWRMultiscale GWRMultiscale.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testGWRMultiscale PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWRMultiscale 
    COMMAND $<TARGET_FILE:testGWRMultiscale> --success
)

add_executable(testLogger Logger.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testLogger PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testLogger 
    COMMAND $<TARGET_FILE:testLogger> --success
)

add_executable(testGWRRobust GWRRobust.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testGWRRobust PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWRRobust 
    COMMAND $<TARGET_FILE:testGWRRobust> --success
)

add_executable(testGWRGeneralized GWRGeneralized.cpp ${SAMPL_LONDONHP100} ${SAMPL_LONDONHP})
target_link_libraries(testGWRGeneralized PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWRGeneralized 
    COMMAND $<TARGET_FILE:testGWRGeneralized> --success
)

add_executable(testGWRScalable GWRScalable.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testGWRScalable PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWRScalable 
    COMMAND $<TARGET_FILE:testGWRScalable> --success
)

add_executable(testLocalCollinearityGWR LocalCollinearityGWR.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testLocalCollinearityGWR PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testLocalCollinearityGWR 
    COMMAND $<TARGET_FILE:testLocalCollinearityGWR> --success
)

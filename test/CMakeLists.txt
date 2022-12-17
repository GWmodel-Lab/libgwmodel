cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Catch2 REQUIRED)

if(ENABLE_OpenMP)
    add_definitions(-DENABLE_OPENMP)
endif()


include_directories(
    ${LIBGWMODEL_INCLUDE_DIR}
    ${ARMADILLO_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(SAMPLE_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)
add_definitions(-DSAMPLE_DATA_DIR="${SAMPLE_DATA_DIR}")
set(SAMPL_LONDONHP100 ${CMAKE_CURRENT_SOURCE_DIR}/data/londonhp100.cpp)

add_executable(testBasicGWR BasicGWR.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testBasicGWR PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2)
add_test(
    NAME testBasicGWR 
    COMMAND $<TARGET_FILE:testBasicGWR> --success
)

add_executable(testGWSS GWSS.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testGWSS PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2)
add_test(
    NAME testGWSS 
    COMMAND $<TARGET_FILE:testGWSS> --success
)

add_executable(testGWPCA GWPCA.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testGWPCA PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2)
add_test(
    NAME testGWPCA 
    COMMAND $<TARGET_FILE:testGWPCA> --success
)

add_executable(testGWDR GWDR.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testGWDR PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2)
add_test(
    NAME testGWDR 
    COMMAND $<TARGET_FILE:testGWDR> --success
)

add_executable(testMGWR MGWR.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testMGWR PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2)
add_test(
    NAME testMGWR 
    COMMAND $<TARGET_FILE:testMGWR> --success
)

add_executable(testLogger Logger.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testLogger PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2)
add_test(
    NAME testLogger 
    COMMAND $<TARGET_FILE:testLogger> --success
)

add_executable(testRobustGWR RobustGWR.cpp ${SAMPL_LONDONHP100})
target_link_libraries(testRobustGWR PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2)
add_test(
    NAME testRobustGWR 
    COMMAND $<TARGET_FILE:testRobustGWR> --success
)

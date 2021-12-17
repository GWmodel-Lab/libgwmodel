cmake_minimum_required(VERSION 3.1)

project(testGWPCA LANGUAGES CXX)

set(SOURCES static.cpp)
if(BUILD_SHARED_LIBS)
set(SOURCES shared.cpp)
endif(BUILD_SHARED_LIBS)

add_executable(testGWPCA ${SOURCES} ${SAMPL_LONDONHP100})
target_link_libraries(testGWPCA PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2)

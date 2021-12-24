cmake_minimum_required(VERSION 3.12)
project(pygwmodel)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})

set(CY_SOURCE 
    ${CMAKE_CURRENT_SOURCE_DIR}/pygwmodel.pyx
)

include_directories(
    ${CMAKE_SOURCE_DIR}/include/gwmodelpp
)

add_cython_target(pygwmodel.pyx CXX)
add_library(pygwmodel MODULE ${pygwmodel})
target_link_libraries(pygwmodel gwmodel ${ARMADILLO_LIBRARIES} ${Python3_LIBRARIES} Python3::NumPy)
python_extension_module(pygwmodel)

install(TARGETS pygwmodel LIBRARY DESTINATION pygwmodel)
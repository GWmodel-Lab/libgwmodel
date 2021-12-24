cmake_minimum_required(VERSION 3.12)


add_cython_target(pygwmodel.pyx CXX)
add_library(pygwmodel MODULE ${pygwmodel})
target_link_libraries(pygwmodel gwmodel ${ARMADILLO_LIBRARIES} ${Python3_LIBRARIES} Python3::NumPy)
python_extension_module(pygwmodel)

install(TARGETS pygwmodel LIBRARY DESTINATION pygwmodel)
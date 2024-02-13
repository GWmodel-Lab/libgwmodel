include(CMakeFindDependencyMacro)
find_dependency(Armadillo)
find_dependency(GSL)
include(${CMAKE_CURRENT_LIST_DIR}/libgwmodelTargets.cmake)
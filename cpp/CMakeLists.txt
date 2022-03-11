cmake_minimum_required(VERSION 3.10)

project(diff_pd)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Pardiso requires stack variables to be aligned.
# Thanks to Fish for figuring out the flag -mpreferred-stack-boundary=4
# This magic number is suitable only for libpardiso600-GNU720-X86-64.so
add_definitions("-Wall -mpreferred-stack-boundary=4")
set(GCC_COVERAGE_COMPILE_FLAGS "-Wl,--no-as-needed")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

# OpenMP.
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    add_definitions(-DEIGEN_DONT_PARALLELIZE)
endif()

# Pardiso.
option(PARDISO_AVAILABLE "Pardiso" OFF)
if(PARDISO_AVAILABLE)
    add_definitions(-DPARDISO_AVAILABLE)
    add_library(pardiso SHARED IMPORTED)
    set_target_properties(pardiso PROPERTIES
        IMPORTED_LOCATION "$ENV{PARDISO_LIC_PATH}/libpardiso600-GNU720-X86-64.so"
    )
else()
    message("Pardiso not detected.")
endif()

# Include directories.
include_directories("../external/eigen")
include_directories("./core/include")
execute_process(COMMAND bash -c "python3-config --includes|sed 's/-I//g'|awk '{print $1;}'" OUTPUT_VARIABLE PYTHON_INCLUDES)
include_directories(${PYTHON_INCLUDES})

# The core library.
file(GLOB_RECURSE CORE_CPP "./core/src/**/*.cpp")
add_library(diff_pd_core SHARED ${CORE_CPP})
if(PARDISO_AVAILABLE)
    target_link_libraries(diff_pd_core pardiso lapack blas OpenMP::OpenMP_CXX pthread m)
else()
    target_link_libraries(diff_pd_core OpenMP::OpenMP_CXX)
endif()

# Python wrapper library.
file(GLOB_RECURSE PYTHON_BINDING "./core/src/**/*.cpp" "./core/src/py_diff_pd_core_wrap.cxx")
add_library(py_diff_pd_core SHARED ${CORE_HEADER} ${PYTHON_BINDING})
if(PARDISO_AVAILABLE)
    target_link_libraries(py_diff_pd_core pardiso lapack blas OpenMP::OpenMP_CXX pthread m)
else()
    target_link_libraries(py_diff_pd_core OpenMP::OpenMP_CXX)
endif()

# Demo.
file(GLOB DEMO_CPP "./main.cpp")
add_executable(diff_pd_demo ${DEMO_CPP})
target_link_libraries(diff_pd_demo diff_pd_core)

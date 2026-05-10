cmake_minimum_required(VERSION 3.10)
project(YANN LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Build options
option(BUILD_SHARED_LIBS "Build project as shared library" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings if pybind11 is available" ON)

# Math backend options
option(USE_EIGEN "Use Eigen library for math operations" ON)
option(USE_OPENBLAS "Use OpenBLAS library for math operations" OFF)
option(USE_MKL "Use Intel MKL library for math operations" OFF)
option(USE_NATIVE "Use native CPU implementation for math operations" OFF)

# Precision options
option(USE_F64 "Use double precision (float 64-bit)" OFF)
option(USE_F32 "Use single precision (float 32-bit)" ON)
option(USE_F16 "Use half precision (float 16-bit)" OFF)
option(USE_BF16 "Use brain float precision (float 16-bit)" OFF)
option(USE_Q8 "Use quarter precision (int 8-bit)" OFF)
# option(USE_DOUBLE_PRECISION "Use double precision (float 64-bit)" OFF)
# option(USE_SINGLE_PRECISION "Use single precision (float 32-bit)" ON)
# option(USE_HALF_PRECISION "Use half precision (float 16-bit)" OFF)
# option(USE_BRAIN_PRECISION "Use brain float precision (float 16-bit)" OFF)
# option(USE_QUARTER_PRECISION "Use quarter precision (int 8-bit)" OFF)

# Check that only one math backend is selected
if((USE_EIGEN AND USE_OPENBLAS) OR (USE_EIGEN AND USE_NATIVE) OR (USE_OPENBLAS AND USE_NATIVE) OR (USE_MKL AND USE_NATIVE))
    message(FATAL_ERROR "Only one math backend can be selected: USE_EIGEN, USE_OPENBLAS, USE_MKL, or USE_NATIVE")
endif()

if(NOT (USE_EIGEN OR USE_OPENBLAS OR USE_NATIVE))
    message(FATAL_ERROR "At least one math backend must be selected: USE_EIGEN, USE_OPENBLAS, or USE_NATIVE")
endif()

# Check that only one precision is selected
set(PRECISION_OPTIONS USE_DOUBLE_PRECISION USE_SINGLE_PRECISION USE_HALF_PRECISION USE_BRAIN_PRECISION USE_QUARTER_PRECISION)
set(SELECTED_PRECISIONS 0)
foreach(OPT IN LISTS PRECISION_OPTIONS)
    if(${OPT})
        math(EXPR SELECTED_PRECISIONS "${SELECTED_PRECISIONS} + 1")
    endif()
endforeach()

if(SELECTED_PRECISIONS EQUAL 0)
    message(FATAL_ERROR "At least one precision must be selected: USE_DOUBLE_PRECISION, USE_SINGLE_PRECISION, USE_HALF_PRECISION, USE_BRAIN_PRECISION, or USE_QUARTER_PRECISION")
endif()

if(SELECTED_PRECISIONS GREATER 1)
    message(FATAL_ERROR "Only one precision can be selected: USE_DOUBLE_PRECISION, USE_SINGLE_PRECISION, USE_HALF_PRECISION, USE_BRAIN_PRECISION, or USE_QUARTER_PRECISION")
endif()

set(YANN_BACKEND_LIBS "")
set(YANN_COMPILE_DEFINITIONS "")

if(USE_EIGEN)
    find_package(Eigen3 5.0 REQUIRED NO_MODULE)
    list(APPEND YANN_BACKEND_LIBS Eigen3::Eigen)
    list(APPEND YANN_COMPILE_DEFINITIONS _USE_EIGEN)
endif()

if(USE_OPENBLAS)
    find_package(BLAS REQUIRED)
    list(APPEND YANN_BACKEND_LIBS BLAS::BLAS)
    list(APPEND YANN_COMPILE_DEFINITIONS _USE_OPENBLAS)
endif()

if(USE_NATIVE_CPU)
    list(APPEND YANN_COMPILE_DEFINITIONS _USE_NATIVE_CPU)
endif()

if(USE_DOUBLE_PRECISION)
    list(APPEND YANN_COMPILE_DEFINITIONS _USE_DOUBLE_PRECISION)
endif()

if(USE_SINGLE_PRECISION)
    list(APPEND YANN_COMPILE_DEFINITIONS _USE_SINGLE_PRECISION)
endif()

if(USE_HALF_PRECISION)
    list(APPEND YANN_COMPILE_DEFINITIONS _USE_HALF_PRECISION)
endif()

if(USE_BRAIN_PRECISION)
    list(APPEND YANN_COMPILE_DEFINITIONS _USE_BRAIN_PRECISION)
endif()

if(USE_QUARTER_PRECISION)
    list(APPEND YANN_COMPILE_DEFINITIONS _USE_QUARTER_PRECISION)
endif()

find_package(OpenMP REQUIRED)
find_package(nlohmann_json 3.12.0 REQUIRED)

add_subdirectory(Cumerics)
add_subdirectory(src)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings/python)
endif()

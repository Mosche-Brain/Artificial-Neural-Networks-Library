cmake_minimum_required(VERSION 3.20)
project(YANN LANGUAGES CXX VERSION 0.1.0)
include(GNUInstallDirs)

# Protect CMAKE_INSTALL_PREFIX from being overwritten by thirdparty subprojects
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation directory" FORCE)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_SHARED_LIBS "Build project as shared library" ON)
option(CUM_BUILD_SHARED "Build CUM library as shared library" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build tests for library" ON)
option(BUILD_TOOLS "Build tools for library" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings if pybind11 is available" OFF)
option(BUILD_C_API "Build the stable C API" OFF)
option(ENABLE_DEBUG_OUTPUT "Enable debug logs" ON)

# Math backend options (handled internally by cum)
option(BUILD_USE_MKL "Use Intel oneMKL library for math operations" ON)
option(BUILD_USE_CBLAS "Use CBLAS library for math operations" OFF)
option(BUILD_USE_NONE "Use native plain C++ implementation for math operations" OFF)

# Compatibility aliases for legacy backend options
if(BUILD_USE_OPENBLAS)
    set(BUILD_USE_CBLAS ON CACHE BOOL "" FORCE)
endif()
if(BUILD_USE_NATIVE)
    set(BUILD_USE_NONE ON CACHE BOOL "" FORCE)
endif()

if(NOT (BUILD_USE_MKL OR BUILD_USE_CBLAS OR BUILD_USE_NONE))
    message(FATAL_ERROR "At least one math backend must be selected: BUILD_USE_MKL, BUILD_USE_CBLAS, or BUILD_USE_NONE")
endif()

# Precision options (aligned with cum module)
option(CUM_USE_FP32 "Use single precision (float 32-bit)" ON)
option(CUM_USE_FP64 "Use double precision (float 64-bit)" OFF)
option(CUM_USE_FP16 "Use half precision (float 16-bit)" OFF)
option(CUM_USE_BF16 "Use brain float precision (float 16-bit)" OFF)
option(CUM_USE_INT8 "Use int8 precision" OFF)

# Backward compatibility mapping for old precision flags
if(CUM_USE_F32)
    set(CUM_USE_FP32 ON CACHE BOOL "" FORCE)
endif()
if(CUM_USE_F64)
    set(CUM_USE_FP64 ON CACHE BOOL "" FORCE)
endif()
if(CUM_USE_F16)
    set(CUM_USE_FP16 ON CACHE BOOL "" FORCE)
endif()
if(CUM_USE_Q8)
    set(CUM_USE_INT8 ON CACHE BOOL "" FORCE)
endif()

# Check that only one precision is selected
set(PRECISION_OPTIONS CUM_USE_FP64 CUM_USE_FP32 CUM_USE_FP16 CUM_USE_BF16 CUM_USE_INT8)
set(SELECTED_PRECISIONS 0)
foreach(OPT IN LISTS PRECISION_OPTIONS)
    if(${OPT})
        math(EXPR SELECTED_PRECISIONS "${SELECTED_PRECISIONS} + 1")
    endif()
endforeach()

if(SELECTED_PRECISIONS EQUAL 0)
    message(WARNING "No precision option selected, defaulting to single precision (float 32-bit)")
    set(CUM_USE_FP32 ON CACHE BOOL "" FORCE)
endif()

if(SELECTED_PRECISIONS GREATER 1)
    message(FATAL_ERROR "Only one precision can be selected: CUM_USE_FP64, CUM_USE_FP32, CUM_USE_FP16, CUM_USE_BF16, or CUM_USE_INT8")
endif()

# Common compiler flags target for internal project components
add_library(yann_compiler_flags INTERFACE)
target_compile_options(yann_compiler_flags INTERFACE
    -Wall
    -Wextra
#    $<$<CXX_COMPILER_ID:Clang,IntelLLVM>:-Wno-deprecated-literal-operator -Wno-macro-redefined>
    -march=native
    $<$<CONFIG:Release>:-O3 -ffast-math>
    $<$<CONFIG:Debug>:-g -O0>
)

find_package(OpenMP REQUIRED)
find_package(nlohmann_json 3.12.0 REQUIRED)

#add_subdirectory(modules/cum)

find_package(cum REQUIRED)

#add_subdirectory(modules/safetensors.cpp)
add_subdirectory(src)

if(BUILD_EXAMPLES)
    add_subdirectory(thirdparty/rapidcsv)
    add_subdirectory(thirdparty/matplotplusplus)
    add_subdirectory(thirdparty/Easy3D)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    add_subdirectory(thirdparty/Catch2)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_TOOLS)
    add_subdirectory(thirdparty/imgui)
    add_subdirectory(tools)
endif()

if(BUILD_C_API)
    add_subdirectory(bindings/C)
endif()

if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings/python)
endif()

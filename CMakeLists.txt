cmake_minimum_required(VERSION 3.28)
project(YANN LANGUAGES CXX VERSION 0.1.0)
include(GNUInstallDirs)
include(cmake/YannCxxModules.cmake)

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
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build tests for library" ON)
option(BUILD_TOOLS "Build tools for library" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings if pybind11 is available" OFF)
option(BUILD_C_API "Build the stable C API" OFF)
option(ENABLE_DEBUG_OUTPUT "Enable debug logs" ON)

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

# Required in the export set because the YANN target links it privately (needed by C++ modules usage requirements)
install(TARGETS yann_compiler_flags EXPORT YANNTargets)

find_package(OpenMP REQUIRED)
find_package(nlohmann_json 3.12.0 REQUIRED)

find_package(cum REQUIRED)

set(SAFETENSORS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SAFETENSORS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(modules/safetensors.cpp)
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

cmake_minimum_required(VERSION 3.14)
project(cum LANGUAGES CXX C VERSION 0.1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(BUILD_USE_MKL "Build CUM library using Intel oneMKL" ON)
option(BUILD_USE_CBLAS "Build CUM library using CBLAS" OFF)
option(BUILD_USE_NONE "Build CUM library using only plain C++" OFF)

set(CUM_BACKEND_COUNT 0)
if(BUILD_USE_MKL)
    math(EXPR CUM_BACKEND_COUNT "${CUM_BACKEND_COUNT}+1")
endif()
if(BUILD_USE_CBLAS)
    math(EXPR CUM_BACKEND_COUNT "${CUM_BACKEND_COUNT}+1")
endif()
if(BUILD_USE_NONE)
    math(EXPR CUM_BACKEND_COUNT "${CUM_BACKEND_COUNT}+1")
endif()
if(CUM_BACKEND_COUNT GREATER 1)
    message(FATAL_ERROR
        "BUILD_USE_MKL, BUILD_USE_CBLAS and BUILD_USE_NONE are mutually exclusive"
    )
endif()

if(NOT BUILD_USE_MKL AND NOT BUILD_USE_CBLAS AND NOT BUILD_USE_NONE)
    set(BUILD_USE_NONE ON)
endif()

# Exactly one numeric type may be selected. FP32 is the default.
option(CUM_USE_FP32 "Use single precision floating point" ON)
option(CUM_USE_FP64 "Use double precision floating point" OFF)
option(CUM_USE_FP16 "Use half precision floating point" OFF)
option(CUM_USE_BF16 "Use bfloat16 precision floating point" OFF)
option(CUM_USE_INT8 "Use int8 precision integer" OFF)

set(CUM_TYPE_COUNT 0)
if(CUM_USE_FP32)
    math(EXPR CUM_TYPE_COUNT "${CUM_TYPE_COUNT}+1")
endif()
if(CUM_USE_FP64)
    math(EXPR CUM_TYPE_COUNT "${CUM_TYPE_COUNT}+1")
endif()
if(CUM_USE_FP16)
    math(EXPR CUM_TYPE_COUNT "${CUM_TYPE_COUNT}+1")
endif()
if(CUM_USE_BF16)
    math(EXPR CUM_TYPE_COUNT "${CUM_TYPE_COUNT}+1")
endif()
if(CUM_USE_INT8)
    math(EXPR CUM_TYPE_COUNT "${CUM_TYPE_COUNT}+1")
endif()
if(CUM_TYPE_COUNT EQUAL 0)
    set(CUM_USE_FP32 ON)
    set(CUM_TYPE_COUNT 1)
endif()
if(CUM_TYPE_COUNT GREATER 1)
    message(FATAL_ERROR
        "CUM_USE_FP32, CUM_USE_FP64, CUM_USE_FP16, CUM_USE_BF16 and CUM_USE_INT8 are mutually exclusive"
    )
endif()

set(CUM_USE_MKL ${BUILD_USE_MKL})

set(CUM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CUM_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(CUM_PROJECT_NAME cum)


configure_file(
    "${CUM_SOURCE_DIR}/config.hpp.in"
    "${CUM_GENERATED_INCLUDE_DIR}/cum/config.hpp"
)

add_subdirectory(sources)

if(BUILD_USE_MKL)
    add_executable(example examples/test.cpp)
    target_link_libraries(example PRIVATE cum::Core cum::MKL)
endif()

cmake_minimum_required(VERSION 3.14)
project(cum LANGUAGES CXX C VERSION 0.1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 23)

set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_C_COMPILER icx)

option(BUILD_USE_MKL "Build CUM library using intel oneMKL" ON)
option(BUILD_USE_CBLAS "Build CUM library cblas" OFF)
option(BUILD_USE_NONE "Build CUM library using only plain C++" OFF)

if(BUILD_USE_MKL)
    set(CMAKE_CXX_COMPILER icpx)
    set(CMAKE_C_COMPILER icx)
endif()

# one option for defining used nummeric type. default is f32, but it can be set to f64 or f16 or bf16 or int8
# option(CUM_USE_F32 "Use single precision floating point" OFF)
# option(CUM_USE_F64 "Use double precision floating point" OFF)
# option(CUM_USE_F16 "Use half precision floating point" OFF)
# option(CUM_USE_BF16 "Use bfloat16 precision floating point" OFF)
# option(CUM_USE_INT8 "Use int8 precision integer" OFF)

# # if everything is off, default to f32
# if(NOT CUM_USE_F32 AND NOT CUM_USE_F64 AND NOT CUM_USE_F16 AND NOT CUM_USE_BF16 AND NOT CUM_USE_INT8)
#     set(CUM_USE_F32 ON)
# endif()

add_subdirectory(sources)

add_executable(example examples/test.cpp)
target_link_libraries(example PRIVATE cum::Core cum::MKL cum_MKL)
target_compile_definitions(example PRIVATE fsycl fPIC)
target_include_directories(example PRIVATE
    /opt/intel/oneapi/compiler/latest/include/    
)

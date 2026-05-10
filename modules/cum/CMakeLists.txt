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

add_subdirectory(sources)

add_executable(example examples/test.cpp)
target_link_libraries(example PRIVATE cum::Core cum::MKL cum_MKL)
target_compile_definitions(example PRIVATE fsycl)
target_include_directories(example PRIVATE
    /opt/intel/oneapi/compiler/latest/include/    
)

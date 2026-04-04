cmake_minimum_required(VERSION 3.18)
project(yann LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Budujemy czystą C++ bibliotekę
add_library(YANN STATIC src/*.cpp)
target_include_directories(YANN PUBLIC include/)
find_package(Eigen3 5.0 REQUIRED NO_MODULE)
target_link_libraries(YANN PUBLIC Eigen3::Eigen)

# 2. Budujemy Python bindings (tylko gdy budujemy z Pythonem)
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    pybind11_add_module(_yann MODULE bindings/python/_yann.cpp)
    target_link_libraries(_yann PRIVATE YANN)
    install(TARGETS _yann LIBRARY DESTINATION yann)
endif()
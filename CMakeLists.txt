cmake_minimum_required(VERSION 3.10)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 23)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(Matplot++ REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)

option(BUILD_SHARED_LIBS "Build project as shared library")

include_directories(
    ./source/
    ./source/Network
    ./source/containers
    ./.external/
    ./.external/imgui
    ./.external/OpenGLPlotLive
    ./.external/imgui/backends
    /usr/include/freetype2
)

file(GLOB_RECURSE PROJECT_SOURCES "source/*.cpp")
file(GLOB_RECURSE NN_LIB_SOURCES "source/Network/*.cpp")

file(GLOB IMGUI_SOURCES
    .external/imgui/*.cpp
    .external/imgui/backends/imgui_impl_glfw.cpp
    .external/imgui/backends/imgui_impl_opengl3.cpp
    .external/implot/*.cpp
)


if(BUILD_SHARED_LIBS)
    add_library(NetworkLib ${NN_LIB_SOURCES})
    set_target_properties(NetworkLib PROPERTIES POSITION_INDEPENDENT_CODE ON SOVERSION 1)
    target_link_libraries(NetworkLib PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
endif()

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} ${IMGUI_SOURCES})
target_compile_options(${PROJECT_NAME} PRIVATE -O3 -ffast-math -fopenmp -march=native)

target_link_libraries(${PROJECT_NAME}
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    OpenGL::GL
    glfw
    GLEW::GLEW
    Matplot++::matplot
    ${CMAKE_DL_LIBS}
)

 
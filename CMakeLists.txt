cmake_minimum_required(VERSION 3.10)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)


# include_directories("./source/")
# include_directories("./source/Network")
# include_directories("./source/containers")
# .external/imgui
# .external/imgui/backends

include_directories(
    ./source/
    ./source/Network
    ./source/containers
    .external/imgui
    .external/imgui/backends
)



file(GLOB_RECURSE SOURCES "source/*.cpp")

file(GLOB IMGUI_SOURCES
    .external/imgui/*.cpp
    .external/imgui/backends/imgui_impl_glfw.cpp
    .external/imgui/backends/imgui_impl_opengl3.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES} ${IMGUI_SOURCES})

target_link_libraries(${PROJECT_NAME}
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    OpenGL::GL
    glfw
    GLEW::GLEW
    ${CMAKE_DL_LIBS}
)

target_include_directories(${PROJECT_NAME} PRIVATE 
    .external/
    .external/imgui 
    .external/imgui/backends
)

target_sources(${PROJECT_NAME} PRIVATE ${IMGUI_SOURCES})

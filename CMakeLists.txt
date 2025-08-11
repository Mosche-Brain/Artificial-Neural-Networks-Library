cmake_minimum_required(VERSION 3.10)
project(NeuralNetworks)

set(CMAKE_CXX_STANDARD 23)

option(BUILD_SHARED_LIBS "Build project as shared library" ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(Matplot++ REQUIRED)

include_directories(
    ./source/
    ./source/Models/
    ./source/Containers/
    ./source/Utils/
)


file(GLOB_RECURSE PROJECT_SOURCES "source/*.cpp")
file(GLOB_RECURSE HEADER_FILES_MODELS "source/Models/*.hpp")
file(GLOB_RECURSE HEADER_FILES_CONTAINERS "source/Containers/*.hpp")
file(GLOB_RECURSE HEADER_FILES_UTILS "source/Utils/*.hpp")

file(COPY ${HEADER_FILES_MODELS} DESTINATION ${CMAKE_BINARY_DIR}/include/Models/)
file(COPY ${HEADER_FILES_CONTAINERS} DESTINATION ${CMAKE_BINARY_DIR}/include/Containers/)
file(COPY ${HEADER_FILES_UTILS} DESTINATION ${CMAKE_BINARY_DIR}/include/Utils/)
file(COPY "source/NeuralNetworksAll.hpp" DESTINATION ${CMAKE_BINARY_DIR}/include/)

add_library(${PROJECT_NAME} SHARED ${PROJECT_SOURCES})
target_compile_options(${PROJECT_NAME} PRIVATE -O3 -ffast-math -fopenmp -march=native)

target_link_libraries(${PROJECT_NAME}
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    Matplot++::matplot
    ${CMAKE_DL_LIBS}
)
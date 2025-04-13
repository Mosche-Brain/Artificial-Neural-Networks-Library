cmake_minimum_required(VERSION 3.10)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nlohmann_json 3.2.0 REQUIRED)

include_directories("./source/")
include_directories("./source/Network")
include_directories("./source/containers")

# Znajdź wszystkie pliki .cpp w katalogu source
file(GLOB_RECURSE SOURCES "source/*.cpp")

# Dodaj pliki źródłowe
add_executable(${PROJECT_NAME} ${SOURCES})

# Połącz z raylib
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} nlohmann_json::nlohmann_json)
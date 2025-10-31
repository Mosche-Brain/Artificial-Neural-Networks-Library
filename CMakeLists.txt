cmake_minimum_required(VERSION 3.10)
project(ANN LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Option to build shared or static library
option(BUILD_SHARED_LIBS "Build project as shared library" ON)

# Find required packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# find_package(nlohmann_json 3.2.0 REQUIRED)
# find_package(Matplot++ REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/Models
    ${CMAKE_SOURCE_DIR}/src/Containers
    ${CMAKE_SOURCE_DIR}/src/Utils
    /usr/includ/Eigen3
    ./src/
)

# Collect src and header files
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS "src/*.hpp" "src/*.h")

# Copy header files to build/include while preserving directory structure
foreach(HEADER ${HEADER_FILES})
    file(RELATIVE_PATH REL_PATH "${CMAKE_SOURCE_DIR}/src" "${HEADER}")
    get_filename_component(REL_DIR "${REL_PATH}" DIRECTORY)
    file(COPY "${HEADER}" DESTINATION "${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/${REL_DIR}")
endforeach()

# Copy NeuralNetworksAll.hpp specifically to build/include
# file(COPY "${CMAKE_SOURCE_DIR}/src/NeuralNetworksAll.hpp" DESTINATION "${CMAKE_BINARY_DIR}/include")
# file(COPY ${CMAKE_SOURCE_DIR}/src/NeuralNetworksAll.hpp DESTINATION ${CMAKE_BINARY_DIR}/include)

# Define the library target
add_library(${PROJECT_NAME} ${PROJECT_SOURCES})

# Set target properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Compiler options
target_compile_options(${PROJECT_NAME} PRIVATE
    #-O3
    #-ffast-math
    -fopenmp
    -march=native
    -Wall
)


# Link libraries
target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    Eigen3::Eigen
    PRIVATE
    ${CMAKE_DL_LIBS})
        
# nlohmann_json::nlohmann_json
# Matplot++::matplot
# Install rules
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install header files while preserving directory structure
install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/
    DESTINATION include
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
)
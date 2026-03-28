cmake_minimum_required(VERSION 3.10)
project(ANN LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Option to build shared or static library
option(BUILD_SHARED_LIBS "Build project as shared library" ON)

# Option to build examples
option(BUILD_EXAMPLES "Build example programs" ON)

# Find required packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)
find_package(sciplot)
find_package(Matplot++ REQUIRED)

# if (OPENMP_FOUND)
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()
   
# find_package(nlohmann_json 3.2.0 REQUIRED)
# find_package(Matplot++ REQUIRED)

# Include directories (prefer target-based includes over global include_directories)
# target_include_directories(${PROJECT_NAME} PUBLIC
include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
    /usr/include/eigen3
)

# Collect source and header files
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS "src/*.hpp" "src/*.h")

# Copy header files to build/include while preserving directory structure
foreach(HEADER ${HEADER_FILES})
    file(RELATIVE_PATH REL_PATH "${CMAKE_SOURCE_DIR}/src" "${HEADER}")
    get_filename_component(REL_DIR "${REL_PATH}" DIRECTORY)
    file(COPY "${HEADER}" DESTINATION "${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/${REL_DIR}")
endforeach()

# Define the library target
add_library(${PROJECT_NAME} ${PROJECT_SOURCES})

# Set target properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    PUBLIC_HEADER "${HEADER_FILES}"
)

# Compiler options
target_compile_options(${PROJECT_NAME} PRIVATE
    -fopenmp
    -march=native
    -Wall
    -O3
    -ffast-math
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
    PRIVATE
        ${CMAKE_DL_LIBS}
)
# nlohmann_json::nlohmann_json
# Matplot++::matplot

# Install rules for the library
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install header files while preserving directory structure
install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/
    DESTINATION include/${PROJECT_NAME}
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
)

# Export targets for find_package support
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)

# === EXAMPLES ===
if(BUILD_EXAMPLES)
    # Find all .cpp files in the example directory
    file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS "examples/*.cpp")

    foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
        # Get filename without extension
        get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)

        # Create executable
        add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})

        # Set C++ standard for example
        set_target_properties(${EXAMPLE_NAME} PROPERTIES
            CXX_STANDARD 23
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
        )


        # Include directories (same as library)
        target_include_directories(${EXAMPLE_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_BINARY_DIR}/include
            /usr/include/eigen3
        )

        # Link with ANN library
        target_link_libraries(${EXAMPLE_NAME} PRIVATE ${PROJECT_NAME}    
                                                    OpenMP::OpenMP_CXX
                                                    sciplot::sciplot
                                                    Matplot++::matplot)

        # Compiler options (same as library)
        target_compile_options(${EXAMPLE_NAME} PRIVATE
            -fopenmp
            -march=native
            -Wall
            -O3
            -ffast-math
        )

        # Optional: install example binaries
        install(TARGETS ${EXAMPLE_NAME}
            RUNTIME DESTINATION build/examples
        )
    endforeach()
endif()
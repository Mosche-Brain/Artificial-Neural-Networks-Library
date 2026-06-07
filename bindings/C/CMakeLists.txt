cmake_minimum_required(VERSION 3.10)

# ============================================================================
# cYANN - Stable C API for YANN
# ============================================================================

if(NOT DEFINED PROJECT_NAME OR NOT TARGET ${PROJECT_NAME})
    message(FATAL_ERROR
        "bindings/C must be built from the main YANN project.\n"
        "Please configure the root project with -DBUILD_C_API=ON")
endif()

# === CRITICAL: Save the real YANN target name BEFORE calling project(cYANN) ===
set(YANN_MAIN_TARGET ${PROJECT_NAME})

project(cYANN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CYANN_TARGET cYANN)

option(YANN_C_API_BUILD_SHARED "Build C API as shared library" ON)

if(YANN_C_API_BUILD_SHARED)
    add_library(${CYANN_TARGET} SHARED)
else()
    add_library(${CYANN_TARGET} STATIC)
endif()

target_sources(${CYANN_TARGET}
    PRIVATE
        src/yann_c_api.cpp
        src/yann_context.cpp
        src/yann_matrix.cpp
        src/yann_model.cpp
)

target_include_directories(${CYANN_TARGET}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/YANN
        ${CMAKE_SOURCE_DIR}/src
)

# ------------------------------------------------------------------
# Linking — use the saved variable, NOT ${PROJECT_NAME}
# ------------------------------------------------------------------
target_link_libraries(${CYANN_TARGET}
    PRIVATE
        ${YANN_MAIN_TARGET}          # ← This is the original "YANN" target
        OpenMP::OpenMP_CXX
        cum::Core
)

# ------------------------------------------------------------------
# SYCL / MKL
# ------------------------------------------------------------------
if(BUILD_USE_MKL)
    target_link_libraries(${CYANN_TARGET}
        PRIVATE
            cum::MKL
            cum_MKL
    )
    target_compile_definitions(${CYANN_TARGET} PRIVATE fsycl fPIC)
    target_compile_options(${CYANN_TARGET} PRIVATE -fsycl)
endif()

target_compile_options(${CYANN_TARGET} PRIVATE
    -qopenmp
    -march=native
    -Wall
    -O3
    -ffast-math
)

if(DEFINED YANN_COMPILE_DEFINITIONS)
    target_compile_definitions(${CYANN_TARGET} PRIVATE ${YANN_COMPILE_DEFINITIONS})
endif()

if(UNIX AND NOT APPLE)
    set_target_properties(${CYANN_TARGET} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()

install(TARGETS ${CYANN_TARGET}
    EXPORT ${CYANN_TARGET}_Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES include/yann.h
    DESTINATION include/yann
)
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS "*.cpp")
file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS "*.hpp" "*.h")

add_library(${PROJECT_NAME} ${PROJECT_SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/yann>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/yann/models>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/yann/models/layers>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/yann/utils>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(${PROJECT_NAME} PUBLIC ${YANN_COMPILE_DEFINITIONS})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    PUBLIC_HEADER "${HEADER_FILES}"
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -qopenmp
    -march=native
    -Wall
    -O3
    -fsycl
    -ffast-math
    -mavxvnni
    -DENABLE_DEBUG_OUTPUT
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        ${YANN_BACKEND_LIBS}
        OpenMP::OpenMP_CXX
    PRIVATE
        nlohmann_json::nlohmann_json
        ${CMAKE_DL_LIBS}
)

# install(TARGETS ${PROJECT_NAME}
#     EXPORT ${PROJECT_NAME}Targets
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
#     RUNTIME DESTINATION bin
#     INCLUDES DESTINATION include
# )

# install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
#     DESTINATION include/
#     FILES_MATCHING
#         PATTERN "*.hpp"
#         PATTERN "*.h"
# )

# install(EXPORT ${PROJECT_NAME}Targets
#     FILE ${PROJECT_NAME}Targets.cmake
#     NAMESPACE ${PROJECT_NAME}::
#     DESTINATION lib/cmake/${PROJECT_NAME}
# )

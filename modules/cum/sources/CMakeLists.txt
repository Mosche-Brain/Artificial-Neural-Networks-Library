add_library(${PROJECT_NAME}_Core STATIC Core.cpp)

target_include_directories(${PROJECT_NAME}_Core
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

add_library(${PROJECT_NAME}::Core ALIAS ${PROJECT_NAME}_Core)

# Backendy
if(BUILD_USE_MKL)
    target_compile_definitions(${PROJECT_NAME}_Core PRIVATE fsycl)
    target_include_directories(${PROJECT_NAME}_Core
        PRIVATE /opt/intel/oneapi/compiler/latest/include/ /opt/intel/oneapi/tbb/latest/include/ /opt/intel/oneapi/compiler/latest/include/
    )
    add_subdirectory(Backends/MKL)
elseif(BUILD_USE_CBLAS)
    add_subdirectory(Backends/CBLAS)
else()
    add_subdirectory(Backends/Native)
endif()
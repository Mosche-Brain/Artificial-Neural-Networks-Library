add_library(${CUM_PROJECT_NAME}_Core STATIC Core.cpp)

target_include_directories(${CUM_PROJECT_NAME}_Core
    PUBLIC
        ${CUM_SOURCE_DIR}/include
        ${CUM_GENERATED_INCLUDE_DIR}
)

add_library(${CUM_PROJECT_NAME}::Core ALIAS ${CUM_PROJECT_NAME}_Core)

if(BUILD_USE_MKL)
    find_path(CUM_SYCL_INCLUDE_DIR sycl/sycl.hpp
        HINTS
            "$ENV{ONEAPI_ROOT}/compiler/latest/include"
            /opt/intel/oneapi/compiler/latest/include
        REQUIRED
    )
    target_compile_definitions(${CUM_PROJECT_NAME}_Core PUBLIC BUILD_USE_MKL)
    target_compile_options(${CUM_PROJECT_NAME}_Core PUBLIC -fsycl)
    target_include_directories(${CUM_PROJECT_NAME}_Core PUBLIC ${CUM_SYCL_INCLUDE_DIR})
    add_subdirectory(Backends/MKL)
elseif(BUILD_USE_CBLAS)
    message(WARNING "CBLAS backend is selected but not implemented; only cum::Core will be built")
elseif(BUILD_USE_NONE)
    message(STATUS "Native backend is selected but not implemented; only cum::Core will be built")
endif()
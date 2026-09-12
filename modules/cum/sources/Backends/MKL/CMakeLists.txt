file(GLOB_RECURSE SOURCES_MKL CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(TBB CONFIG REQUIRED)
find_package(MKL CONFIG REQUIRED)
find_package(dnnl CONFIG REQUIRED)
find_package(oneDPL CONFIG REQUIRED)

add_library(${CUM_PROJECT_NAME}_MKL STATIC
    ${SOURCES_MKL}
)

target_include_directories(${CUM_PROJECT_NAME}_MKL
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${CUM_PROJECT_NAME}_MKL
    PUBLIC
        ${CUM_PROJECT_NAME}::Core
        TBB::tbb
        MKL::MKL
        MKL::MKL_DPCPP
        oneDPL
        DNNL::dnnl
)

target_compile_definitions(${CUM_PROJECT_NAME}_MKL
    PUBLIC BUILD_USE_MKL
)

target_compile_options(${CUM_PROJECT_NAME}_MKL
    PUBLIC -fsycl
)

target_link_options(${CUM_PROJECT_NAME}_MKL
    PUBLIC -fsycl
)

add_library(${CUM_PROJECT_NAME}::MKL ALIAS ${CUM_PROJECT_NAME}_MKL)

file(GLOB_RECURSE SOURCES_MKL CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(TBB CONFIG REQUIRED)
find_package(MKL CONFIG REQUIRED)
find_package(dnnl CONFIG REQUIRED
    PATHS "${CUM_ONEDNN_INSTALL_DIR}/lib/cmake/dnnl"
    NO_DEFAULT_PATH
)
find_package(oneDPL CONFIG REQUIRED)

add_library(${CUM_PROJECT_NAME}_MKL ${CUM_LIBRARY_TYPE}
    ${SOURCES_MKL}
)

target_include_directories(${CUM_PROJECT_NAME}_MKL
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(${CUM_PROJECT_NAME}_MKL
    PUBLIC BUILD_USE_MKL
)

if(CUM_BUILD_SHARED)
    target_compile_options(${CUM_PROJECT_NAME}_MKL PRIVATE -fsycl)
    target_link_options(${CUM_PROJECT_NAME}_MKL PRIVATE -fsycl)
    target_link_libraries(${CUM_PROJECT_NAME}_MKL
        PUBLIC
            ${CUM_PROJECT_NAME}::Core
        PRIVATE
            TBB::tbb
            MKL::MKL
            MKL::MKL_DPCPP
            oneDPL
            DNNL::dnnl
    )
    set_target_properties(${CUM_PROJECT_NAME}_MKL PROPERTIES
        INSTALL_RPATH "$ORIGIN:${CMAKE_INSTALL_FULL_LIBDIR}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
else()
    target_compile_options(${CUM_PROJECT_NAME}_MKL PUBLIC -fsycl)
    target_link_options(${CUM_PROJECT_NAME}_MKL PUBLIC -fsycl)
    target_link_libraries(${CUM_PROJECT_NAME}_MKL
        PUBLIC
            ${CUM_PROJECT_NAME}::Core
            TBB::tbb
            MKL::MKL
            MKL::MKL_DPCPP
            oneDPL
            DNNL::dnnl
    )
endif()

add_library(${CUM_PROJECT_NAME}::MKL ALIAS ${CUM_PROJECT_NAME}_MKL)

install(TARGETS ${CUM_PROJECT_NAME}_MKL
    EXPORT cumTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(TARGET DNNL::dnnl)
    install(DIRECTORY "${CUM_ONEDNN_INSTALL_DIR}/lib/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING
            PATTERN "libdnnl.so*"
    )
endif()

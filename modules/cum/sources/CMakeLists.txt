include(GNUInstallDirs)

add_library(${CUM_PROJECT_NAME}_Core ${CUM_LIBRARY_TYPE} Core.cpp)

target_include_directories(${CUM_PROJECT_NAME}_Core
        PUBLIC
        $<BUILD_INTERFACE:${CUM_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CUM_GENERATED_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_library(${CUM_PROJECT_NAME}::Core ALIAS ${CUM_PROJECT_NAME}_Core)

set_target_properties(${CUM_PROJECT_NAME}_Core PROPERTIES EXPORT_NAME Core)

if(BUILD_USE_MKL)
    target_compile_definitions(${CUM_PROJECT_NAME}_Core PUBLIC BUILD_USE_MKL)

    if(CUM_BUILD_SHARED)
        target_compile_options(${CUM_PROJECT_NAME}_Core PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-fsycl>)
        target_link_options(${CUM_PROJECT_NAME}_Core PRIVATE -fsycl)
        set_target_properties(${CUM_PROJECT_NAME}_Core PROPERTIES
            INSTALL_RPATH "$ORIGIN:${CMAKE_INSTALL_FULL_LIBDIR}"
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    else()
        target_compile_options(${CUM_PROJECT_NAME}_Core PUBLIC
                $<$<COMPILE_LANGUAGE:CXX>:-fsycl>)
        target_link_options(${CUM_PROJECT_NAME}_Core PUBLIC -fsycl)
    endif()

    add_subdirectory(Backends/MKL)
elseif(BUILD_USE_CBLAS)
    message(WARNING "CBLAS backend is selected but not implemented; only cum::Core will be built")
elseif(BUILD_USE_NONE)
    message(STATUS "Native backend is selected but not implemented; only cum::Core will be built")
endif()

add_library(${CUM_PROJECT_NAME} INTERFACE)
add_library(${CUM_PROJECT_NAME}::${CUM_PROJECT_NAME} ALIAS ${CUM_PROJECT_NAME})
target_link_libraries(${CUM_PROJECT_NAME} INTERFACE ${CUM_PROJECT_NAME}::Core)

if(BUILD_USE_MKL)
    target_link_libraries(${CUM_PROJECT_NAME} INTERFACE ${CUM_PROJECT_NAME}::MKL)
endif()

install(TARGETS ${CUM_PROJECT_NAME}_Core ${CUM_PROJECT_NAME}
        EXPORT cumTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY ${CUM_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CUM_GENERATED_INCLUDE_DIR}/cum/config.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cum)

install(EXPORT cumTargets
        FILE cumTargets.cmake
        NAMESPACE cum::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cum
)

include(CMakePackageConfigHelpers)
set(CUM_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/cum")

configure_package_config_file(
    "${CUM_SOURCE_DIR}/cmake/cumConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cumConfig.cmake"
    INSTALL_DESTINATION "${CUM_CONFIG_INSTALL_DIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cumConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cumConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cumConfigVersion.cmake"
    DESTINATION "${CUM_CONFIG_INSTALL_DIR}"
)
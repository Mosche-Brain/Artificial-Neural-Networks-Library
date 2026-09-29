include(GNUInstallDirs)

add_library(${CUM_PROJECT_NAME}_Core STATIC Core.cpp)

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

    target_compile_options(${CUM_PROJECT_NAME}_Core PUBLIC
            $<$<COMPILE_LANGUAGE:CXX>:-fsycl>)
    target_link_options(${CUM_PROJECT_NAME}_Core PUBLIC -fsycl)

    add_subdirectory(Backends/MKL)
elseif(BUILD_USE_CBLAS)
    message(WARNING "CBLAS backend is selected but not implemented; only cum::Core will be built")
elseif(BUILD_USE_NONE)
    message(STATUS "Native backend is selected but not implemented; only cum::Core will be built")
endif()

install(TARGETS ${CUM_PROJECT_NAME}_Core
        EXPORT cumTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY ${CUM_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CUM_GENERATED_INCLUDE_DIR}/cum/config.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cum)
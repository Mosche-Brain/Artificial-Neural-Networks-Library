file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)

    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})

    target_include_directories(${EXAMPLE_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(${EXAMPLE_NAME} PRIVATE
        ${PROJECT_NAME}::${PROJECT_NAME}
        matplot
        yann_compiler_flags
    )

    if(EXAMPLE_NAME STREQUAL "mnist")
        find_package(ZLIB REQUIRED)
        target_link_libraries(${EXAMPLE_NAME} PRIVATE ZLIB::ZLIB)
    endif()
endforeach()

file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
)

foreach(EXAMPLE_SOURCE IN LISTS EXAMPLE_SOURCES)
    get_filename_component(EXAMPLE_NAME "${EXAMPLE_SOURCE}" NAME_WE)
    set(EXAMPLE_TARGET "cyann_example_${EXAMPLE_NAME}")

    add_executable(${EXAMPLE_TARGET} "${EXAMPLE_SOURCE}")

    set_target_properties(${EXAMPLE_TARGET} PROPERTIES
        C_STANDARD 17
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
        OUTPUT_NAME ${EXAMPLE_NAME}
    )

    target_link_libraries(${EXAMPLE_TARGET} PRIVATE cYANN)
endforeach()

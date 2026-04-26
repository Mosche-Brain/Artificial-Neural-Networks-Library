file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)

    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})

    set_target_properties(${EXAMPLE_NAME} PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    target_include_directories(${EXAMPLE_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        /usr/include/eigen3
    )

    target_link_libraries(${EXAMPLE_NAME} PRIVATE ${PROJECT_NAME} OpenMP::OpenMP_CXX)

    target_compile_options(${EXAMPLE_NAME} PRIVATE
        -fopenmp
        -march=native
        -Wall
        -O3
        -ffast-math
    )

    install(TARGETS ${EXAMPLE_NAME}
        RUNTIME DESTINATION build/examples
    )
endforeach()

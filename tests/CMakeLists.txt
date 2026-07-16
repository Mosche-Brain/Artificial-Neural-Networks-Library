list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
include(Catch)

file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/*.cpp)

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
    )

    target_link_libraries(${EXAMPLE_NAME} PRIVATE ${PROJECT_NAME}
            OpenMP::OpenMP_CXX
            cum::Core
            cum::MKL
            Catch2::Catch2WithMain)

    catch_discover_tests(${EXAMPLE_NAME})

    # if using MKL, link to it
    if(BUILD_USE_MKL)
        target_link_libraries(${EXAMPLE_NAME} PRIVATE cum::MKL cum_MKL)
        target_compile_definitions(${EXAMPLE_NAME} PRIVATE fsycl fPIC)
        target_compile_options(${EXAMPLE_NAME} PRIVATE -fsycl)
    endif()

    target_compile_options(${EXAMPLE_NAME} PRIVATE
            -qopenmp
            -march=native
            -Wall
            -O3
            -ffast-math
            # ${YANN_COMPILE_DEFINITIONS}
    )

    target_compile_definitions(${EXAMPLE_NAME} PRIVATE ${YANN_COMPILE_DEFINITIONS})

    install(TARGETS ${EXAMPLE_NAME}
            RUNTIME DESTINATION build/tests
    )
endforeach()
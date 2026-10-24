list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
include(Catch)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    set(EXEC_NAME ${TEST_NAME})
    if(TARGET ${TEST_NAME})
        set(EXEC_NAME "test_${TEST_NAME}")
    endif()

    add_executable(${EXEC_NAME} ${TEST_SOURCE})

    target_include_directories(${EXEC_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(${EXEC_NAME} PRIVATE
        ${PROJECT_NAME}::${PROJECT_NAME}
        Catch2::Catch2WithMain
        yann_compiler_flags
    )

    catch_discover_tests(${EXEC_NAME})
endforeach()
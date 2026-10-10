find_package(pybind11 CONFIG QUIET)

if(pybind11_FOUND)
    pybind11_add_module(_yann MODULE _yann.cpp)
    target_link_libraries(_yann PRIVATE ${PROJECT_NAME}::${PROJECT_NAME} yann_compiler_flags)
    install(TARGETS _yann LIBRARY DESTINATION yann)
endif()
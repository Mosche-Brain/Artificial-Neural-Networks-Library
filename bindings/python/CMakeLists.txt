find_package(pybind11 CONFIG QUIET)

if(pybind11_FOUND)
    pybind11_add_module(_yann MODULE _yann.cpp)
    target_link_libraries(_yann PRIVATE YANN)
    install(TARGETS _yann LIBRARY DESTINATION yann)
endif()
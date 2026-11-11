# CMake has no built-in C++ modules support for the IntelLLVM compiler id (icpx),
# although icpx is Clang-based. This module bridges clang-scan-deps and compilation.

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    return()
endif()

get_filename_component(_yann_cxx_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
find_program(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS
    NAMES clang-scan-deps
    HINTS "${_yann_cxx_dir}" "${_yann_cxx_dir}/compiler"
)
if(NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
    message(FATAL_ERROR "C++ modules with ${CMAKE_CXX_COMPILER} require clang-scan-deps")
endif()

execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-resource-dir
    OUTPUT_VARIABLE _yann_resource_dir OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_CXX_SCANDEP_SOURCE
    "\"${CMAKE_CURRENT_LIST_DIR}/icpx-scan-deps.sh\" \"${CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS}\" \"${_yann_resource_dir}\""
    " -- <CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -x c++ -c -o <OBJECT>"
    " -MT <DYNDEP_FILE> -MD -MF <DEP_FILE> <SOURCE>"
    " > <DYNDEP_FILE>.tmp && mv <DYNDEP_FILE>.tmp <DYNDEP_FILE>")
string(CONCAT CMAKE_CXX_SCANDEP_SOURCE ${CMAKE_CXX_SCANDEP_SOURCE})
set(CMAKE_CXX_MODULE_MAP_FORMAT "clang")
set(CMAKE_CXX_MODULE_MAP_FLAG "@<MODULE_MAP_FILE>")
set(CMAKE_CXX_COMPILE_BMI
    "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> --precompile <SOURCE>")
set(CMAKE_CXX_MODULE_EXTENSION ".pcm")

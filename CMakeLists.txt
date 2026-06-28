cmake_minimum_required(VERSION 3.10)
project(YANN LANGUAGES CXX)

add_compile_options(-Wno-deprecated-literal-operator -Wno-macro-redefined)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Build options
option(BUILD_SHARED_LIBS "Build project as shared library" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings if pybind11 is available" OFF)
option(BUILD_C_API "Build the stable C API" ON)
option(ENABLE_DEBUG_OUTPUT "Enable debug logs" ON)

# Math backend options
option(BUILD_USE_EIGEN "Use Eigen library for math operations" OFF)
option(BUILD_USE_OPENBLAS "Use OpenBLAS library for math operations" OFF)
option(BUILD_USE_MKL "Use Intel oneMKL library for math operations" ON)
option(BUILD_USE_NATIVE "Use native C++ implementation for math operations" OFF)

# Precision options
option(CUM_USE_F64 "Use double precision (float 64-bit)" OFF)
option(CUM_USE_F32 "Use single precision (float 32-bit)" OFF)
option(CUM_USE_F16 "Use half precision (float 16-bit)" OFF)
option(CUM_USE_BF16 "Use brain float precision (float 16-bit)" OFF)
option(CUM_USE_Q8 "Use quarter precision (int 8-bit)" OFF)

if(NOT (BUILD_USE_EIGEN OR BUILD_USE_OPENBLAS OR BUILD_USE_NATIVE OR BUILD_USE_MKL))
    message(FATAL_ERROR "At least one math backend must be selected: USE_EIGEN, USE_OPENBLAS, or USE_NATIVE")
endif()

# Check that only one precision is selected
set(PRECISION_OPTIONS CUM_USE_F64 CUM_USE_F32 CUM_USE_F16 CUM_USE_BF16 CUM_USE_Q8)
set(SELECTED_PRECISIONS 0)
foreach(OPT IN LISTS PRECISION_OPTIONS)
    if(${OPT})
        math(EXPR SELECTED_PRECISIONS "${SELECTED_PRECISIONS} + 1")
    endif()
endforeach()

if(SELECTED_PRECISIONS EQUAL 0)
    # message(FATAL_ERROR "At least one precision must be selected: USE_DOUBLE_PRECISION, USE_SINGLE_PRECISION, USE_HALF_PRECISION, USE_BRAIN_PRECISION, or USE_QUARTER_PRECISION")
    message(WARNING "No precision option selected, defaulting to single precision (float 32-bit)")
    set(CUM_USE_F32 ON)
endif()

if(SELECTED_PRECISIONS GREATER 1)
    message(FATAL_ERROR "Only one precision can be selected: USE_DOUBLE_PRECISION, USE_SINGLE_PRECISION, USE_HALF_PRECISION, USE_BRAIN_PRECISION, or USE_QUARTER_PRECISION")
endif()

set(YANN_BACKEND_LIBS "")
set(YANN_COMPILE_DEFINITIONS "")

if(USE_EIGEN)
    find_package(Eigen3 5.0 REQUIRED NO_MODULE)
    list(APPEND YANN_BACKEND_LIBS Eigen3::Eigen)
    list(APPEND YANN_COMPILE_DEFINITIONS _USE_EIGEN)
endif()

if(USE_OPENBLAS)
    find_package(BLAS REQUIRED)
    list(APPEND YANN_BACKEND_LIBS BLAS::BLAS)
    list(APPEND YANN_COMPILE_DEFINITIONS _USE_OPENBLAS)
endif()

if(BUILD_USE_MKL)
    # set(CMAKE_CXX_COMPILER /opt/intel/oneapi/compiler/latest/bin/icpx)
    # set(CMAKE_C_COMPILER /opt/intel/oneapi/compiler/latest/bin/icx)
    list(APPEND YANN_COMPILE_DEFINITIONS BUILD_USE_MKL)
    list(APPEND YANN_COMPILE_DEFINITIONS fsycl)
    list(APPEND YANN_BACKEND_LIBS cum::Core)
    list(APPEND YANN_BACKEND_LIBS cum::MKL)
endif()

if(USE_NATIVE)
    list(APPEND YANN_COMPILE_DEFINITIONS CUM_USE_NATIVE)
endif()

if(CUM_USE_F64)
    list(APPEND YANN_COMPILE_DEFINITIONS CUM_USE_F64)
endif()

if(CUM_USE_F32)
    list(APPEND YANN_COMPILE_DEFINITIONS CUM_USE_F32)
endif()

if(CUM_USE_F16)
    list(APPEND YANN_COMPILE_DEFINITIONS CUM_USE_F16)
endif()

if(CUM_USE_BF16)
    list(APPEND YANN_COMPILE_DEFINITIONS CUM_USE_BF16)
endif()

if(CUM_USE_Q8)
    list(APPEND YANN_COMPILE_DEFINITIONS CUM_USE_Q8)
endif()

find_package(OpenMP REQUIRED)
find_package(nlohmann_json 3.12.0 REQUIRED)

# Przekaż definicje precyzji do wszystkich celów (w tym moduły cum)
if(CUM_USE_F64)
    add_compile_definitions(CUM_USE_F64)
endif()
if(CUM_USE_F32)
    add_compile_definitions(CUM_USE_F32)
endif()
if(CUM_USE_F16)
    add_compile_definitions(CUM_USE_F16)
endif()
if(CUM_USE_BF16)
    add_compile_definitions(CUM_USE_BF16)
endif()
if(CUM_USE_Q8)
    add_compile_definitions(CUM_USE_Q8)
endif()

add_subdirectory(modules/cum)
add_subdirectory(src)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
    add_subdirectory(thirdparty/rapidcsv)
    add_subdirectory(thirdparty/matplotplusplus)
endif()

if(BUILD_C_API)
    add_subdirectory(bindings/C)
endif()

if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings/python)
endif()


file(GLOB_RECURSE SOURCES_MKL *.cpp *.c)

set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_C_COMPILER icx)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(dnnl_DIR "/home/jaro/Development/Yann/thirdparty/oneDNN/install/lib/cmake/dnnl/" CACHE PATH "" FORCE)

find_package(TBB CONFIG REQUIRED PATHS $ENV{TBBROOT} /opt/intel/oneapi/tbb/latest/lib/cmake/tbb/)
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT} /opt/intel/oneapi/mkl/latest/lib/cmake/mkl/)
find_package(dnnl CONFIG REQUIRED NO_DEFAULT_PATH)
find_package(oneDPL CONFIG REQUIRED)

add_library(${PROJECT_NAME}_MKL STATIC
    ${SOURCES_MKL}
)

target_include_directories(${PROJECT_NAME}_MKL PRIVATE
    /opt/intel/oneapi/compiler/latest/include/ /opt/intel/oneapi/tbb/latest/include/ /opt/intel/oneapi/compiler/latest/include/ ${CMAKE_SOURCE_DIR}/modules/cum/sources/Backends/MKL/
)

target_link_libraries(${PROJECT_NAME}_MKL 
    PRIVATE ${PROJECT_NAME}::Core 
    TBB::tbb
    MKL::MKL
    MKL::MKL_DPCPP
    oneDPL
    DNNL::dnnl
)

target_compile_definitions(${PROJECT_NAME}_MKL 
    PRIVATE fsycl fPIC
)
target_compile_options(${PROJECT_NAME}_MKL
    PUBLIC -g
)

add_library(${PROJECT_NAME}::MKL ALIAS ${PROJECT_NAME}_MKL)

# target_compile_definitions(${PROJECT_NAME}_Core PUBLIC BUILD_USE_MKL fsycl)

file(GLOB_RECURSE SOURCES_MKL *.cpp *.c)

set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_C_COMPILER icx)

set(CMAKE_CXX_STANDARD 23)

find_package(TBB CONFIG REQUIRED PATHS $ENV{TBBROOT} /opt/intel/oneapi/tbb/latest/lib/cmake/tbb/)
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT} /opt/intel/oneapi/mkl/latest/lib/cmake/mkl/)

add_library(${PROJECT_NAME}_MKL STATIC
    ${SOURCES_MKL}
)

target_include_directories(${PROJECT_NAME}_MKL PRIVATE
    /opt/intel/oneapi/compiler/latest/include/ /opt/intel/oneapi/tbb/latest/include/ /opt/intel/oneapi/compiler/latest/include/  
)

target_link_libraries(${PROJECT_NAME}_MKL 
    PRIVATE ${PROJECT_NAME}::Core 
    TBB::tbb
    # TBB::TBB_DPCPP
    MKL::MKL
    MKL::MKL_DPCPP
)

target_compile_definitions(${PROJECT_NAME}_MKL 
    PRIVATE fsycl
)

add_library(${PROJECT_NAME}::MKL ALIAS ${PROJECT_NAME}_MKL)

# target_compile_definitions(${PROJECT_NAME}_Core PUBLIC BUILD_USE_MKL fsycl)
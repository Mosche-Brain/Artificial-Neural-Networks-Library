add_library(cum_C STATIC
    src/cum.cpp
)

set_target_properties(cum_C PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(cum_C
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(TARGET cum::MKL)
    target_link_libraries(cum_C PUBLIC cum::MKL)
else()
    target_link_libraries(cum_C PUBLIC cum::Core)
endif()

add_library(cum::C ALIAS cum_C)

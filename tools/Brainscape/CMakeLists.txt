add_executable(Brainscape
    main.cpp
    Window.cpp
    Gui.cpp
    ModelSession.cpp
    NetworkVisualizer.cpp
    Application.cpp
)

set_target_properties(Brainscape PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_include_directories(Brainscape PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(Brainscape PRIVATE
    ${PROJECT_NAME}
    imgui::imgui
    OpenMP::OpenMP_CXX
    cum::Core
    cum::MKL
)

if(BUILD_USE_MKL)
    target_link_libraries(Brainscape PRIVATE cum::MKL cum_MKL)
    target_compile_definitions(Brainscape PRIVATE fsycl fPIC)
    target_compile_options(Brainscape PRIVATE -fsycl)
endif()

target_compile_options(Brainscape PRIVATE
    -qopenmp
    -march=native
    -Wall
    -O3
    -ffast-math
)

target_compile_definitions(Brainscape PRIVATE ${YANN_COMPILE_DEFINITIONS})

install(TARGETS Brainscape
    RUNTIME DESTINATION build/tools
)

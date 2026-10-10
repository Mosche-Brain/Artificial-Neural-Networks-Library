add_executable(Brainscape
    main.cpp
    Window.cpp
    Gui.cpp
    ModelSession.cpp
    NetworkVisualizer.cpp
    Application.cpp
)

target_include_directories(Brainscape PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(Brainscape PRIVATE
    ${PROJECT_NAME}::${PROJECT_NAME}
    imgui::imgui
    yann_compiler_flags
)

install(TARGETS Brainscape
    RUNTIME DESTINATION build/tools
)

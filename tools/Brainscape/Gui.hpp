#pragma once

#include "Window.hpp"

class Gui
{
public:
    explicit Gui(Window& window);
    ~Gui();

    Gui(const Gui&) = delete;
    Gui& operator=(const Gui&) = delete;

    void beginFrame();
    void endFrame();

private:
    Window& window_;
};

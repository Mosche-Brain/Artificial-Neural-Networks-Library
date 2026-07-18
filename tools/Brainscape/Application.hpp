#pragma once

#include "Gui.hpp"
#include "ModelSession.hpp"
#include "NetworkVisualizer.hpp"
#include "Window.hpp"

#include "imgui.h"

class Application
{
public:
    enum class ViewMode
    {
        Network = 0,
        Curve = 1,
        Split = 2,
    };

    Application();
    ~Application() = default;

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;

    int run();

private:
    void drawUi(const NetworkSnapshot& snap);
    void drawControlPanel(const NetworkSnapshot& snap, bool training, int nEval);
    void drawViewport(const NetworkSnapshot& snap, const ImVec2& origin, const ImVec2& size);

    Window window_;
    Gui gui_;
    ModelSession session_;
    NetworkVisualizer visualizer_;

    std::size_t sampleIndex_ = 0;
    bool animate_ = true;
    float learningRate_ = 0.01f;
    int epochs_ = 100;
    int lossIndex_ = 0;
    ViewMode viewMode_ = ViewMode::Split;
};

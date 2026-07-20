#include "Application.hpp"

#include "runtime_config.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

#include "imgui.h"

#include <yann/optimizers/SGD.hpp>

namespace
{
    yann::models::Sequential makeDefaultModel()
    {
        // yann::optimizers::SGD::create(0.01);

        return yann::models::Sequential({
            yann::models::layers::Input::createUnique(1),
            yann::models::layers::Dense::createUnique(20, "tanh"),
            yann::models::layers::Dense::createUnique(20, "tanh"),
            // yann::models::layers::Dense::createUnique(8, "leaky_relu"),
            // yann::models::layers::Dense::createUnique(16, "leaky_relu"),
            // yann::models::layers::Dense::createUnique(8, "leaky_relu"),
            // yann::models::layers::Dense::createUnique(8, "leaky_relu"),
            // yann::models::layers::Dense::createUnique(8, "leaky_relu"),
            // yann::models::layers::Dense::createUnique(8, "leaky_relu"),
            // yann::models::layers::Dense::createUnique(8, "leaky_relu"),
            // yann::models::layers::Dense::createUnique(8, "leaky_relu"),
            // yann::models::layers::Dense::createUnique(24, "leaky_relu"),
            // yann::models::layers::Dense::createUnique(16, "tanh"),
            // yann::models::layers::Dense::createUnique(32, "sigmoid"),
            // yann::models::layers::Dense::createUnique(16, "tanh"),
            // yann::models::layers::Dense::createUnique(32, "tanh"),
            yann::models::layers::Dense::createUnique(1, "linear"),
        });
    }

    yann::optimizers::Optimizer makeDefaultOptimizer()
    {
        return yann::optimizers::SGD::create(0.01);
    }

    void setupSessionData(ModelSession& session)
    {
        const float x_min = -4.0f * static_cast<float>(M_PI);
        const float x_max =  4.0f * static_cast<float>(M_PI);
        const std::size_t N_train = 64;
        const std::size_t N_eval = 512;

        // Column samples (N x 1). Build Y explicitly as sin(X) — do not reuse X buffer.
        cum::Matrix X_train(N_train, 1);
        cum::Matrix Y_train(N_train, 1);
        for (std::size_t i = 0; i < N_train; ++i)
        {
            const float t = (N_train == 1)
                ? x_min
                : x_min + static_cast<float>(i) * (x_max - x_min) / static_cast<float>(N_train - 1);
            X_train(i, 0) = static_cast<cum::cumeric_t>(t);
            Y_train(i, 0) = static_cast<cum::cumeric_t>(std::sin(t));
        }

        cum::Matrix X_eval(N_eval, 1);
        for (std::size_t i = 0; i < N_eval; ++i)
        {
            const float t = (N_eval == 1)
                ? x_min
                : x_min + static_cast<float>(i) * (x_max - x_min) / static_cast<float>(N_eval - 1);
            X_eval(i, 0) = static_cast<cum::cumeric_t>(t);
        }

        // divide x_eval and x_train by x_max
        X_train /= x_max;
        X_eval /= x_max;

        session.setTrainingData(std::move(X_train), std::move(Y_train));
        session.setEvalInputs(std::move(X_eval));
    }

    constexpr float kSidePanelWidth = 300.0f;
    constexpr float kPanelGap = 10.0f;
}

Application::Application()
    : window_(1920, 1080, "Brainscape")
    , gui_(window_)
    , session_(makeDefaultModel(), makeDefaultOptimizer())
{
    yann::runtime_config::set_verbosity(0);
    setupSessionData(session_);
}

void Application::drawControlPanel(const NetworkSnapshot& snap, bool training, int nEval)
{
    ImGuiIO& io = ImGui::GetIO();

    static constexpr const char* kLossNames[] = {
        "mse",
        "cross_entropy",
        "binary_cross_entropy",
    };
    static constexpr yann::utils::loss::LossFunction kLossValues[] = {
        yann::utils::loss::LossFunction::mse,
        yann::utils::loss::LossFunction::cross_entropy,
        yann::utils::loss::LossFunction::binary_cross_entropy,
    };

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.07f, 0.07f, 0.08f, 1.0f));
    ImGui::BeginChild("Controls", ImVec2(kSidePanelWidth, 0.0f), ImGuiChildFlags_Borders);

    ImGui::TextUnformatted("BRAINSCAPE");
    ImGui::Separator();

    ImGui::Spacing();
    ImGui::TextUnformatted("View");
    int view = static_cast<int>(viewMode_);
    if (ImGui::RadioButton("Network", view == static_cast<int>(ViewMode::Network)))
        viewMode_ = ViewMode::Network;
    ImGui::SameLine();
    if (ImGui::RadioButton("Curve", view == static_cast<int>(ViewMode::Curve)))
        viewMode_ = ViewMode::Curve;
    ImGui::SameLine();
    if (ImGui::RadioButton("Split", view == static_cast<int>(ViewMode::Split)))
        viewMode_ = ViewMode::Split;

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextUnformatted("Model");
    ImGui::BeginDisabled(training);
    if (ImGui::Button("Reset", ImVec2(-1.0f, 0.0f)))
        session_.reset(makeDefaultModel());
    ImGui::EndDisabled();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextUnformatted("Training");
    ImGui::BeginDisabled(training);
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::InputFloat("##rate", &learningRate_, 0.001f, 0.01f, "rate: %.4f");
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::InputInt("##epochs", &epochs_);
    if (epochs_ < 1)
        epochs_ = 1;
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::Combo("##loss", &lossIndex_, kLossNames, IM_ARRAYSIZE(kLossNames)))
        session_.setLossFunction(kLossValues[lossIndex_]);
    ImGui::EndDisabled();

    if (training)
    {
        ImGui::BeginDisabled();
        ImGui::Button("Training...", ImVec2(-1.0f, 36.0f));
        ImGui::EndDisabled();
    }
    else if (ImGui::Button("Train", ImVec2(-1.0f, 36.0f)))
    {
        session_.setLossFunction(kLossValues[lossIndex_]);
        session_.startTraining(static_cast<cum::cumeric_t>(learningRate_),
                               static_cast<std::size_t>(epochs_));
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextUnformatted("Probe");
    ImGui::Checkbox("Animate input", &animate_);
    int sample = static_cast<int>(sampleIndex_);
    if (nEval > 0)
    {
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::SliderInt("##sample", &sample, 0, nEval - 1, "sample: %d"))
        {
            sampleIndex_ = static_cast<std::size_t>(sample);
            animate_ = false;
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextUnformatted("Status");
    ImGui::Text("x_in  %.4f", snap.x);
    ImGui::Text("y_out %.4f", snap.yHat);
    ImGui::Text("FPS   %.1f", io.Framerate);
    if (training)
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.2f, 1.0f), "training...");
    else
        ImGui::TextDisabled("idle");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextUnformatted("Weight");
    if (const auto& h = visualizer_.hoveredWeight())
    {
        ImGui::Text("hover L%zu  %d -> %d", h->layer, h->fromNeuron, h->toNeuron);
        ImGui::Text("w = %.6f", h->weight);
    }
    else if (const auto& s = visualizer_.selectedWeight())
    {
        ImGui::Text("pin   L%zu  %d -> %d", s->layer, s->fromNeuron, s->toNeuron);
        ImGui::Text("w = %.6f", s->weight);
        if (ImGui::SmallButton("Clear pin"))
            visualizer_.clearSelection();
    }
    else
    {
        ImGui::TextDisabled("hover an edge");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextWrapped("Hover edge = weight. Click to pin. Nodes: activation. Curve: train vs model.");

    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void Application::drawViewport(const NetworkSnapshot& snap, const ImVec2& origin, const ImVec2& size)
{
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(origin,
                            ImVec2(origin.x + size.x, origin.y + size.y),
                            IM_COL32(0, 0, 0, 255));

    const ResponseCurve& curve = session_.responseCurve();
    constexpr float gap = 8.0f;
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const ImVec2* mousePtr = &mouse;

    switch (viewMode_)
    {
    case ViewMode::Network:
        visualizer_.draw(drawList, origin, size, snap, mousePtr);
        break;
    case ViewMode::Curve:
        visualizer_.drawResponseCurve(drawList, origin, size, curve);
        break;
    case ViewMode::Split:
    default:
    {
        const float curveH = std::clamp(size.y * 0.34f, 160.0f, 320.0f);
        const float netH = std::max(80.0f, size.y - curveH - gap);
        visualizer_.draw(drawList, origin, ImVec2(size.x, netH), snap, mousePtr);
        visualizer_.drawResponseCurve(drawList,
                                      ImVec2(origin.x, origin.y + netH + gap),
                                      ImVec2(size.x, curveH),
                                      curve);
        break;
    }
    }
}

void Application::drawUi(const NetworkSnapshot& snap)
{
    ImGuiIO& io = ImGui::GetIO();
    const bool training = session_.isTraining();
    const int nEval = static_cast<int>(session_.evalSampleCount());

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(kPanelGap, kPanelGap));
    ImGui::Begin("Brainscape", nullptr,
                 ImGuiWindowFlags_NoDecoration |
                 ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoBringToFrontOnFocus |
                 ImGuiWindowFlags_NoResize);

    drawControlPanel(snap, training, nEval);
    ImGui::SameLine(0.0f, kPanelGap);

    ImGui::BeginChild("Viewport", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders);
    const ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    const ImVec2 canvasSize = ImGui::GetContentRegionAvail();
    drawViewport(snap, canvasPos, canvasSize);
    ImGui::Dummy(canvasSize);
    ImGui::EndChild();

    ImGui::End();
    ImGui::PopStyleVar();
}

int Application::run()
{
    using clock = std::chrono::steady_clock;
    constexpr auto frameDuration = std::chrono::duration<double>(1.0 / 30.0);

    while (!window_.shouldClose())
    {
        const auto frameStart = clock::now();

        window_.pollEvents();
        gui_.beginFrame();

        const std::size_t nEval = session_.evalSampleCount();
        if (animate_ && nEval > 0)
            sampleIndex_ = (sampleIndex_ + 1) % nEval;

        const NetworkSnapshot snap = session_.updateSnapshot(sampleIndex_);
        drawUi(snap);

        gui_.endFrame();
        window_.swapBuffers();

        const auto elapsed = clock::now() - frameStart;
        if (elapsed < frameDuration)
            std::this_thread::sleep_for(frameDuration - elapsed);
    }
    return 0;
}

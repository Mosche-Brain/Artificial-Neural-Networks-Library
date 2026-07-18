#pragma once

#include "NetworkSnapshot.hpp"
#include "ResponseCurve.hpp"

#include "imgui.h"

#include <optional>

struct WeightHoverInfo
{
    std::size_t layer = 0;   // destination dense layer index
    int fromNeuron = 0;      // previous layer
    int toNeuron = 0;        // this layer
    float weight = 0.0f;
};

class NetworkVisualizer
{
public:
    void draw(ImDrawList* drawList, const ImVec2& origin, const ImVec2& size,
              const NetworkSnapshot& snap, const ImVec2* mousePos = nullptr);

    void drawResponseCurve(ImDrawList* drawList, const ImVec2& origin, const ImVec2& size,
                           const ResponseCurve& curve) const;

    const std::optional<WeightHoverInfo>& hoveredWeight() const { return hovered_; }
    const std::optional<WeightHoverInfo>& selectedWeight() const { return selected_; }
    void clearSelection() { selected_.reset(); }

private:
    static ImVec4 valueToRgb(float value, float maxAbs);
    static ImU32 valueToColor(float value, float maxAbs, float alpha = 1.0f);
    static ImU32 darkenColor(ImVec4 color, float factor = 0.35f, float alpha = 0.55f);
    static float distPointToSegment(const ImVec2& p, const ImVec2& a, const ImVec2& b);

    std::optional<WeightHoverInfo> hovered_;
    std::optional<WeightHoverInfo> selected_;
};

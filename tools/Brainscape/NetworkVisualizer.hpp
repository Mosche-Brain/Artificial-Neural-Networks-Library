#pragma once

#include "NetworkSnapshot.hpp"
#include "ResponseCurve.hpp"

#include "imgui.h"

class NetworkVisualizer
{
public:
    void draw(ImDrawList* drawList, const ImVec2& origin, const ImVec2& size,
              const NetworkSnapshot& snap) const;

    void drawResponseCurve(ImDrawList* drawList, const ImVec2& origin, const ImVec2& size,
                           const ResponseCurve& curve) const;

private:
    static ImVec4 valueToRgb(float value, float maxAbs);
    static ImU32 valueToColor(float value, float maxAbs, float alpha = 1.0f);
    static ImU32 darkenColor(ImVec4 color, float factor = 0.35f, float alpha = 0.55f);
};

#include "NetworkVisualizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

ImVec4 NetworkVisualizer::valueToRgb(float value, float maxAbs)
{
    const float scale = maxAbs > 1e-8f ? maxAbs : 1.0f;
    const float t = 0.5f * (std::clamp(value / scale, -1.0f, 1.0f) + 1.0f);
    return ImVec4(1.0f - t, t, 0.0f, 1.0f);
}

ImU32 NetworkVisualizer::valueToColor(float value, float maxAbs, float alpha)
{
    ImVec4 color = valueToRgb(value, maxAbs);
    color.w = alpha;
    return ImGui::ColorConvertFloat4ToU32(color);
}

ImU32 NetworkVisualizer::darkenColor(ImVec4 color, float factor, float alpha)
{
    color.x *= factor;
    color.y *= factor;
    color.z *= factor;
    color.w = alpha;
    return ImGui::ColorConvertFloat4ToU32(color);
}

void NetworkVisualizer::draw(ImDrawList* drawList, const ImVec2& origin, const ImVec2& size,
                             const NetworkSnapshot& snap) const
{
    const std::size_t layers = snap.layerSizes.size();
    if (!snap.valid || layers == 0 || size.x < 32.0f || size.y < 32.0f)
        return;

    int maxNeurons = 1;
    for (int s : snap.layerSizes)
        maxNeurons = std::max(maxNeurons, s);

    const float padX = 48.0f;
    const float padY = 36.0f;
    const float usableW = size.x - 2.0f * padX;
    const float usableH = size.y - 2.0f * padY;
    const float layerGap = layers > 1 ? usableW / static_cast<float>(layers - 1) : 0.0f;
    const float nodeRadius = std::clamp(usableH / static_cast<float>(maxNeurons * 3), 6.0f, 18.0f);

    auto nodePos = [&](std::size_t layer, int neuron) -> ImVec2
    {
        const float x = origin.x + padX + static_cast<float>(layer) * layerGap;
        const float count = static_cast<float>(snap.layerSizes[layer]);
        const float span = usableH * (count > 1.0f ? 1.0f : 0.0f);
        const float step = count > 1.0f ? span / (count - 1.0f) : 0.0f;
        const float y0 = origin.y + padY + (usableH - span) * 0.5f;
        return ImVec2(x, y0 + static_cast<float>(neuron) * step);
    };

    float weightMax = 1e-3f;
    for (std::size_t layer = 1; layer < layers; ++layer)
        for (const auto& row : snap.weights[layer])
            for (float w : row)
                weightMax = std::max(weightMax, std::fabs(w));

    float outputMax = 1e-3f;
    for (const auto& layerOuts : snap.outputs)
        for (float v : layerOuts)
            outputMax = std::max(outputMax, std::fabs(v));

    for (std::size_t layer = 1; layer < layers; ++layer)
    {
        const int inCount = snap.layerSizes[layer - 1];
        const int outCount = snap.layerSizes[layer];
        const auto& W = snap.weights[layer];

        for (int out = 0; out < outCount; ++out)
        {
            for (int in = 0; in < inCount; ++in)
            {
                if (out >= static_cast<int>(W.size()) ||
                    in >= static_cast<int>(W[static_cast<std::size_t>(out)].size()))
                    continue;
                const float w = W[static_cast<std::size_t>(out)][static_cast<std::size_t>(in)];
                const float thickness = 0.6f + 2.4f * (std::fabs(w) / weightMax);
                drawList->AddLine(nodePos(layer - 1, in), nodePos(layer, out),
                                  valueToColor(w, weightMax, 0.55f), thickness);
            }
        }
    }

    for (std::size_t layer = 0; layer < layers; ++layer)
    {
        for (int n = 0; n < snap.layerSizes[layer]; ++n)
        {
            const float value = n < static_cast<int>(snap.outputs[layer].size())
                ? snap.outputs[layer][static_cast<std::size_t>(n)]
                : 0.0f;
            const ImVec2 p = nodePos(layer, n);
            const ImVec4 rgb = valueToRgb(value, outputMax);
            drawList->AddCircleFilled(p, nodeRadius, darkenColor(rgb), 24);
            drawList->AddCircle(p, nodeRadius, ImGui::ColorConvertFloat4ToU32(rgb), 24, 2.0f);
        }
    }
}

void NetworkVisualizer::drawResponseCurve(ImDrawList* drawList, const ImVec2& origin,
                                          const ImVec2& size, const ResponseCurve& curve) const
{
    if (size.x < 64.0f || size.y < 64.0f)
        return;

    drawList->AddRectFilled(origin, ImVec2(origin.x + size.x, origin.y + size.y),
                            IM_COL32(0, 0, 0, 255));
    drawList->AddRect(origin, ImVec2(origin.x + size.x, origin.y + size.y),
                      IM_COL32(60, 60, 60, 255));

    const float padL = 48.0f;
    const float padR = 16.0f;
    const float padT = 28.0f;
    const float padB = 32.0f;
    const ImVec2 plotMin(origin.x + padL, origin.y + padT);
    const ImVec2 plotMax(origin.x + size.x - padR, origin.y + size.y - padB);
    const float plotW = plotMax.x - plotMin.x;
    const float plotH = plotMax.y - plotMin.y;
    if (plotW < 8.0f || plotH < 8.0f)
        return;

    float xMin = 0.0f, xMax = 1.0f, yMin = -1.0f, yMax = 1.0f;
    bool hasData = false;
    auto expand = [&](float x, float y)
    {
        if (!hasData)
        {
            xMin = xMax = x;
            yMin = yMax = y;
            hasData = true;
            return;
        }
        xMin = std::min(xMin, x);
        xMax = std::max(xMax, x);
        yMin = std::min(yMin, y);
        yMax = std::max(yMax, y);
    };

    for (std::size_t i = 0; i < curve.xs.size() && i < curve.ys.size(); ++i)
        expand(curve.xs[i], curve.ys[i]);
    for (std::size_t i = 0; i < curve.trainXs.size() && i < curve.trainYs.size(); ++i)
        expand(curve.trainXs[i], curve.trainYs[i]);
    if (curve.valid)
        expand(curve.markerX, curve.markerY);

    if (!hasData)
        return;

    const float xPad = std::max(0.05f * (xMax - xMin), 1e-3f);
    const float yPad = std::max(0.05f * (yMax - yMin), 1e-3f);
    xMin -= xPad;
    xMax += xPad;
    yMin -= yPad;
    yMax += yPad;

    auto toScreen = [&](float x, float y) -> ImVec2
    {
        const float u = (x - xMin) / (xMax - xMin);
        const float v = (y - yMin) / (yMax - yMin);
        return ImVec2(plotMin.x + u * plotW, plotMax.y - v * plotH);
    };

    drawList->AddLine(ImVec2(plotMin.x, plotMax.y), ImVec2(plotMax.x, plotMax.y),
                      IM_COL32(120, 120, 120, 200), 1.0f);
    drawList->AddLine(ImVec2(plotMin.x, plotMin.y), ImVec2(plotMin.x, plotMax.y),
                      IM_COL32(120, 120, 120, 200), 1.0f);

    if (yMin < 0.0f && yMax > 0.0f)
    {
        const ImVec2 z0 = toScreen(xMin, 0.0f);
        const ImVec2 z1 = toScreen(xMax, 0.0f);
        drawList->AddLine(z0, z1, IM_COL32(70, 70, 70, 180), 1.0f);
    }

    for (std::size_t i = 0; i < curve.trainXs.size() && i < curve.trainYs.size(); ++i)
    {
        const ImVec2 p = toScreen(curve.trainXs[i], curve.trainYs[i]);
        drawList->AddCircleFilled(p, 3.0f, IM_COL32(180, 180, 80, 200), 8);
    }

    if (curve.xs.size() >= 2 && curve.xs.size() == curve.ys.size())
    {
        for (std::size_t i = 1; i < curve.xs.size(); ++i)
        {
            drawList->AddLine(toScreen(curve.xs[i - 1], curve.ys[i - 1]),
                              toScreen(curve.xs[i], curve.ys[i]),
                              IM_COL32(80, 200, 120, 255), 2.0f);
        }
    }

    if (curve.valid)
    {
        const ImVec2 m = toScreen(curve.markerX, curve.markerY);
        drawList->AddCircleFilled(m, 5.0f, IM_COL32(255, 80, 80, 255), 12);
        drawList->AddCircle(m, 5.0f, IM_COL32(255, 220, 220, 255), 12, 1.5f);
        drawList->AddLine(ImVec2(m.x, plotMin.y), ImVec2(m.x, plotMax.y),
                          IM_COL32(255, 80, 80, 80), 1.0f);
    }

    char label[64];
    std::snprintf(label, sizeof(label), "y_out vs x_in");
    drawList->AddText(ImVec2(plotMin.x, origin.y + 6.0f), IM_COL32(200, 200, 200, 255), label);

    std::snprintf(label, sizeof(label), "%.2f", xMin);
    drawList->AddText(ImVec2(plotMin.x, plotMax.y + 4.0f), IM_COL32(150, 150, 150, 255), label);
    std::snprintf(label, sizeof(label), "%.2f", xMax);
    drawList->AddText(ImVec2(plotMax.x - 40.0f, plotMax.y + 4.0f), IM_COL32(150, 150, 150, 255), label);
    std::snprintf(label, sizeof(label), "%.2f", yMax);
    drawList->AddText(ImVec2(origin.x + 4.0f, plotMin.y), IM_COL32(150, 150, 150, 255), label);
    std::snprintf(label, sizeof(label), "%.2f", yMin);
    drawList->AddText(ImVec2(origin.x + 4.0f, plotMax.y - 14.0f), IM_COL32(150, 150, 150, 255), label);
}

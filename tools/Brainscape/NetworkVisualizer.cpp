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

float NetworkVisualizer::distPointToSegment(const ImVec2& p, const ImVec2& a, const ImVec2& b)
{
    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    const float len2 = dx * dx + dy * dy;
    if (len2 < 1e-12f)
    {
        const float ex = p.x - a.x;
        const float ey = p.y - a.y;
        return std::sqrt(ex * ex + ey * ey);
    }
    float t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2;
    t = std::clamp(t, 0.0f, 1.0f);
    const float px = a.x + t * dx - p.x;
    const float py = a.y + t * dy - p.y;
    return std::sqrt(px * px + py * py);
}

void NetworkVisualizer::draw(ImDrawList* drawList, const ImVec2& origin, const ImVec2& size,
                             const NetworkSnapshot& snap, const ImVec2* mousePos)
{
    hovered_.reset();

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

    const bool mouseIn =
        mousePos &&
        mousePos->x >= origin.x && mousePos->x <= origin.x + size.x &&
        mousePos->y >= origin.y && mousePos->y <= origin.y + size.y;

    float bestDist = 8.0f;
    WeightHoverInfo best{};
    bool foundHover = false;

    if (mouseIn)
    {
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
                    const ImVec2 a = nodePos(layer - 1, in);
                    const ImVec2 b = nodePos(layer, out);
                    const float d = distPointToSegment(*mousePos, a, b);
                    if (d < bestDist)
                    {
                        bestDist = d;
                        best = WeightHoverInfo{
                            layer, in, out,
                            W[static_cast<std::size_t>(out)][static_cast<std::size_t>(in)]
                        };
                        foundHover = true;
                    }
                }
            }
        }
    }

    if (foundHover)
        hovered_ = best;

    if (mouseIn && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
    {
        if (foundHover)
            selected_ = best;
        else
            selected_.reset();
    }

    // keep pinned weight value in sync with current snapshot
    if (selected_ && selected_->layer < snap.weights.size())
    {
        const auto& W = snap.weights[selected_->layer];
        if (selected_->toNeuron >= 0 &&
            selected_->toNeuron < static_cast<int>(W.size()) &&
            selected_->fromNeuron >= 0 &&
            selected_->fromNeuron < static_cast<int>(W[static_cast<std::size_t>(selected_->toNeuron)].size()))
        {
            selected_->weight = W[static_cast<std::size_t>(selected_->toNeuron)]
                                  [static_cast<std::size_t>(selected_->fromNeuron)];
        }
    }

    auto isFocus = [&](std::size_t layer, int in, int out) -> bool
    {
        if (hovered_ && hovered_->layer == layer && hovered_->fromNeuron == in && hovered_->toNeuron == out)
            return true;
        if (selected_ && selected_->layer == layer && selected_->fromNeuron == in && selected_->toNeuron == out)
            return true;
        return false;
    };

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
                const bool focus = isFocus(layer, in, out);
                const ImU32 color = focus
                    ? IM_COL32(255, 255, 255, 255)
                    : valueToColor(w, weightMax, 0.55f);
                drawList->AddLine(nodePos(layer - 1, in), nodePos(layer, out),
                                  color, focus ? thickness + 2.0f : thickness);
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

    const WeightHoverInfo* info = hovered_ ? &*hovered_ : (selected_ ? &*selected_ : nullptr);
    if (info)
    {
        const ImVec2 a = nodePos(info->layer - 1, info->fromNeuron);
        const ImVec2 b = nodePos(info->layer, info->toNeuron);
        const ImVec2 mid((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f);

        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "W[L%zu][%d<-%d] = %.6f",
                      info->layer, info->toNeuron, info->fromNeuron, info->weight);

        const ImVec2 textSize = ImGui::CalcTextSize(buf);
        const ImVec2 pad(8.0f, 6.0f);
        const ImVec2 boxMin(mid.x - textSize.x * 0.5f - pad.x, mid.y - textSize.y - pad.y - 10.0f);
        const ImVec2 boxMax(boxMin.x + textSize.x + pad.x * 2.0f, boxMin.y + textSize.y + pad.y * 2.0f);

        drawList->AddRectFilled(boxMin, boxMax, IM_COL32(20, 20, 24, 230), 4.0f);
        drawList->AddRect(boxMin, boxMax, IM_COL32(220, 220, 220, 200), 4.0f);
        drawList->AddText(ImVec2(boxMin.x + pad.x, boxMin.y + pad.y),
                          IM_COL32(255, 255, 255, 255), buf);

        if (hovered_)
            ImGui::SetTooltip("layer %zu  |  neuron %d -> %d\nweight = %.8f\n(click to pin)",
                              info->layer, info->fromNeuron, info->toNeuron, info->weight);
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
    const float padT = 40.0f;
    const float padB = 36.0f;
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
    {
        expand(curve.trainXs[i], curve.trainYs[i]);
        if (i < curve.trainPredYs.size())
            expand(curve.trainXs[i], curve.trainPredYs[i]);
    }
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

    const std::size_t nTrain = std::min({curve.trainXs.size(), curve.trainYs.size(), curve.trainPredYs.size()});
    for (std::size_t i = 0; i < nTrain; ++i)
    {
        const ImVec2 pTarget = toScreen(curve.trainXs[i], curve.trainYs[i]);
        const ImVec2 pPred = toScreen(curve.trainXs[i], curve.trainPredYs[i]);
        drawList->AddLine(pTarget, pPred, IM_COL32(255, 90, 90, 160), 1.5f);
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

    for (std::size_t i = 0; i < nTrain; ++i)
    {
        const ImVec2 pTarget = toScreen(curve.trainXs[i], curve.trainYs[i]);
        const ImVec2 pPred = toScreen(curve.trainXs[i], curve.trainPredYs[i]);
        drawList->AddCircleFilled(pPred, 3.5f, IM_COL32(80, 200, 120, 230), 10);
        drawList->AddCircle(pPred, 3.5f, IM_COL32(40, 120, 70, 255), 10, 1.0f);
        drawList->AddCircleFilled(pTarget, 4.0f, IM_COL32(230, 200, 60, 240), 10);
        drawList->AddCircle(pTarget, 4.0f, IM_COL32(255, 240, 160, 255), 10, 1.2f);
    }

    if (curve.trainPredYs.empty())
    {
        for (std::size_t i = 0; i < curve.trainXs.size() && i < curve.trainYs.size(); ++i)
        {
            const ImVec2 p = toScreen(curve.trainXs[i], curve.trainYs[i]);
            drawList->AddCircleFilled(p, 4.0f, IM_COL32(230, 200, 60, 240), 10);
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

    char label[96];
    std::snprintf(label, sizeof(label), "train vs model  |  MSE %.5f", curve.mse);
    drawList->AddText(ImVec2(plotMin.x, origin.y + 6.0f), IM_COL32(200, 200, 200, 255), label);

    const float legendY = origin.y + 22.0f;
    float lx = plotMin.x;
    drawList->AddCircleFilled(ImVec2(lx + 5.0f, legendY + 6.0f), 4.0f, IM_COL32(230, 200, 60, 255), 8);
    drawList->AddText(ImVec2(lx + 14.0f, legendY), IM_COL32(200, 200, 200, 255), "target");
    lx += 78.0f;
    drawList->AddLine(ImVec2(lx, legendY + 7.0f), ImVec2(lx + 18.0f, legendY + 7.0f),
                      IM_COL32(80, 200, 120, 255), 2.0f);
    drawList->AddCircleFilled(ImVec2(lx + 9.0f, legendY + 7.0f), 3.0f, IM_COL32(80, 200, 120, 255), 8);
    drawList->AddText(ImVec2(lx + 24.0f, legendY), IM_COL32(200, 200, 200, 255), "model");
    lx += 86.0f;
    drawList->AddLine(ImVec2(lx, legendY + 3.0f), ImVec2(lx, legendY + 11.0f),
                      IM_COL32(255, 90, 90, 200), 1.5f);
    drawList->AddText(ImVec2(lx + 8.0f, legendY), IM_COL32(200, 200, 200, 255), "error");

    std::snprintf(label, sizeof(label), "%.2f", xMin);
    drawList->AddText(ImVec2(plotMin.x, plotMax.y + 4.0f), IM_COL32(150, 150, 150, 255), label);
    std::snprintf(label, sizeof(label), "%.2f", xMax);
    drawList->AddText(ImVec2(plotMax.x - 40.0f, plotMax.y + 4.0f), IM_COL32(150, 150, 150, 255), label);
    std::snprintf(label, sizeof(label), "%.2f", yMax);
    drawList->AddText(ImVec2(origin.x + 4.0f, plotMin.y), IM_COL32(150, 150, 150, 255), label);
    std::snprintf(label, sizeof(label), "%.2f", yMin);
    drawList->AddText(ImVec2(origin.x + 4.0f, plotMax.y - 14.0f), IM_COL32(150, 150, 150, 255), label);
}

#include <cum/Tensor.hpp>
#include <cum/Vector.hpp>
#include <cum/cum.hpp>
#include <cum/Matrix.hpp>
#include <cum/runtime.hpp>

#include <matplot/matplot.h>

#include <algorithm>
#include <print>
#include <cstdint>
#include <zlib.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <numeric>
#include <array>

#include "helpers/MnistDataLoader.hpp"

import yann;

using helpers::MnistDataLoader;

using namespace yann::models;

#include <raylib.h>
#include <vector>
#include <cmath>
#include <algorithm>


void raylib_demo(Sequential& model)
{
    constexpr int GRID_SIZE = 28;
    constexpr int CELL_SCALE = 16;
    constexpr int CANVAS_SIZE = GRID_SIZE * CELL_SCALE; // 448 px
    constexpr int PADDING = 20;

    constexpr int SCREEN_WIDTH = CANVAS_SIZE + 380 + PADDING * 3; // 888 px
    constexpr int SCREEN_HEIGHT = CANVAS_SIZE + PADDING * 2 + 40; // 528 px

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "YANN - MNIST Interactive Demo");
    SetTargetFPS(60);

    std::vector<float> canvas(GRID_SIZE * GRID_SIZE, 0.0f);
    std::vector<float> probabilities(10, 0.0f);
    int top_digit = -1;
    bool dirty = true;

    auto run_inference = [&]() {
        // Tensor wejściowy dla sieci: batch 1, 784 cechy
        cum::Tensor input(cum::Shape{1, GRID_SIZE * GRID_SIZE});

        for (int64_t i = 0; i < GRID_SIZE * GRID_SIZE; ++i) {
            input.at({0, i}) = canvas[i];
        }

        // Forward przez Sequential
        cum::Tensor output = model.forward(input);

        top_digit = 0;
        float max_prob = -1.0f;
        for (int64_t i = 0; i < 10; ++i) {
            // Bezpieczny odczyt wartości przez at({0, i}) lub at({i})
            float prob = static_cast<float>(output.at({0, i}));
            probabilities[i] = prob;
            if (prob > max_prob) {
                max_prob = prob;
                top_digit = static_cast<int>(i);
            }
        }
    };

    while (!WindowShouldClose())
    {
        Vector2 mouse = GetMousePosition();
        bool left_down = IsMouseButtonDown(MOUSE_BUTTON_LEFT);
        bool right_down = IsMouseButtonDown(MOUSE_BUTTON_RIGHT);

        // Obsługa rysowania na płótnie
        if (mouse.x >= PADDING && mouse.x < PADDING + CANVAS_SIZE &&
            mouse.y >= PADDING && mouse.y < PADDING + CANVAS_SIZE)
        {
            if (left_down || right_down) {
                float gx = (mouse.x - PADDING) / CELL_SCALE;
                float gy = (mouse.y - PADDING) / CELL_SCALE;
                float brush_radius = 1.35f;

                int min_x = std::max(0, static_cast<int>(gx - brush_radius));
                int max_x = std::min(GRID_SIZE - 1, static_cast<int>(gx + brush_radius));
                int min_y = std::max(0, static_cast<int>(gy - brush_radius));
                int max_y = std::min(GRID_SIZE - 1, static_cast<int>(gy + brush_radius));

                for (int y = min_y; y <= max_y; ++y) {
                    for (int x = min_x; x <= max_x; ++x) {
                        float dist = std::hypot(x + 0.5f - gx, y + 0.5f - gy);
                        if (dist < brush_radius) {
                            float factor = (brush_radius - dist) / brush_radius;
                            int idx = y * GRID_SIZE + x;
                            if (left_down) {
                                canvas[idx] = std::clamp(canvas[idx] + factor * 0.45f, 0.0f, 1.0f);
                            } else {
                                canvas[idx] = std::clamp(canvas[idx] - factor * 0.70f, 0.0f, 1.0f);
                            }
                            dirty = true;
                        }
                    }
                }
            }
        }

        if (IsKeyPressed(KEY_C)) {
            std::fill(canvas.begin(), canvas.end(), 0.0f);
            dirty = true;
        }

        if (dirty) {
            run_inference();
            dirty = false;
        }

        BeginDrawing();
        ClearBackground(Color{24, 26, 31, 255});

        // 1. Siatka pikseli 28x28
        for (int y = 0; y < GRID_SIZE; ++y) {
            for (int x = 0; x < GRID_SIZE; ++x) {
                unsigned char val = static_cast<unsigned char>(canvas[y * GRID_SIZE + x] * 255.0f);
                DrawRectangle(
                    PADDING + x * CELL_SCALE,
                    PADDING + y * CELL_SCALE,
                    CELL_SCALE,
                    CELL_SCALE,
                    Color{val, val, val, 255}
                );
            }
        }
        DrawRectangleLines(PADDING - 1, PADDING - 1, CANVAS_SIZE + 2, CANVAS_SIZE + 2, DARKGRAY);

        // 2. Słupki prawdopodobieństw klas (0-9)
        const int panel_x = PADDING * 2 + CANVAS_SIZE;
        const int bar_max_width = 200;
        DrawText("Predictions", panel_x, PADDING, 22, WHITE);

        for (int i = 0; i < 10; ++i) {
            int row_y = PADDING + 45 + i * 36;
            bool is_best = (i == top_digit && probabilities[i] > 0.15f);

            DrawText(TextFormat("%d:", i), panel_x, row_y + 2, 20, is_best ? GREEN : LIGHTGRAY);

            // Tło paska
            DrawRectangle(panel_x + 30, row_y, bar_max_width, 22, Color{45, 48, 55, 255});

            // Wypełnienie
            int filled_w = static_cast<int>(probabilities[i] * bar_max_width);
            Color bar_col = is_best ? Color{50, 205, 50, 255} : Color{70, 130, 180, 255};
            DrawRectangle(panel_x + 30, row_y, filled_w, 22, bar_col);

            // Wartość %
            DrawText(TextFormat("%5.1f%%", probabilities[i] * 100.0f), panel_x + 30 + bar_max_width + 10, row_y + 3, 16, RAYWHITE);
        }

        // Najbardziej prawdopodobna cyfra
        if (top_digit >= 0 && probabilities[top_digit] > 0.05f) {
            DrawText(
                TextFormat("Digit: %d (%2.0f%%)", top_digit, probabilities[top_digit] * 100.0f),
                panel_x,
                PADDING + 45 + 10 * 36 + 10,
                24,
                GREEN
            );
        }

        DrawText("LPM: Rysuj  |  PPM: Gumka  |  [C]: Wyczyść", PADDING, CANVAS_SIZE + PADDING + 12, 16, GRAY);

        EndDrawing();
    }

    CloseWindow();
}

int main(int argc, char** argv)
{
    cum::cum(cum::DEVICE::CPU);

    const std::string mnist_directory = argc > 1 ? argv[1] : "/home/jaro/Development/Yann/Assets/mnist";

    auto train = MnistDataLoader::load(mnist_directory, "train");
    auto test = MnistDataLoader::load(mnist_directory, "test");

    train.images /= static_cast<cum::cumeric_t>(255);
    test.images /= static_cast<cum::cumeric_t>(255);

    std::println("MNIST loaded");
    std::println("training samples: {} | test samples: {}", train.images.rows(), test.images.rows());
    std::println("Images: {}x{} | {}", train.images.rows(), train.images.cols(), train.images(9, 9));
    std::println("Labels: {}x{}", train.labels.rows(), train.labels.cols());

    Sequential model({
        layers::Input::createUnique(28 * 28),
        layers::Dense::createUnique(256, "gelu"),
        layers::Dense::createUnique(128, "gelu"),
        layers::Dense::createUnique(64, "tanh"),
        layers::Linear::createUnique(10),
        layers::Softmax::createUnique(10, 1)
    });

    yann::loss::Loss loss = yann::loss::CategoricalCrossEntropy::create();
    yann::optimizers::Optimizer optimizer = yann::optimizers::ADAM::create(0.001f);

    yann::logging::LossTracker loss_tracker;
    std::array<yann::logging::ITrainingCallback*, 1> callbacks = {
        &loss_tracker
    };

	yann::runtime_config::set_verbosity(1);

    // yann::utils::model_deserialize_from_safetensors(model, "mnist-resolver-adam.safetensors");

    constexpr std::size_t epochs = 8;
    constexpr std::size_t batch_size = 128;
    model.fit(train.images, train.labels, *loss, *optimizer, epochs, batch_size, callbacks);

    yann::utils::model_serialize_to_safetensors(model, "mnist-resolver-adam-2.safetensors");

    raylib_demo(model);

    const auto loss_history = loss_tracker.getLossHistory();
    std::vector<double> epoch_range(loss_history.size());
    std::iota(epoch_range.begin(), epoch_range.end(), 0.0);

    auto figure = matplot::figure(true);
    matplot::plot(epoch_range, loss_history);
    matplot::title("MNIST training loss");
    matplot::xlabel("epoch");
    matplot::ylabel("loss");
    figure->size(1200, 800);
    matplot::show();

    cum::decum();
    return 0;
}

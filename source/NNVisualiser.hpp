#pragma once
#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "./imgui/imgui.h"
#include "./imgui/backends/imgui_impl_glfw.h"
#include "./imgui/backends/imgui_impl_opengl3.h"

#include "implot/implot.h"

#include <thread>
#include <mutex>
#include <atomic>

#include "Network/NeuralNet.hpp"

class NNVisualiser
{
public:
    NNVisualiser(NeuralNet* network);
    ~NNVisualiser();

    void Render();
    void display();
    void playGraph();
    
    // float inputs[2] = {0.0f, 0.0f};
    float training_epochs = 1000.0f;
    float learning_rate = 0.1f;
    std::string forward_result = "N/A";
    std::string training_status = "Untrained";

    MatrixXd train_x;
                    
    VectorXd train_y;

protected:
    void drawNeuron(glm::vec2 pos, float radius);
    void drawConnection(glm::vec2 from, glm::vec2 to, float weight);
    void drawCircle(float cx, float cy, float r, int num_segments);

    void startTraining(int _iter_, float _rate_);
    void stopTraining();

    GLFWwindow* window;
    std::thread thread;
    NeuralNet* network;

    std::mutex network_mutex;
    std::atomic<bool> is_training{false};

    float min_range_x = -10.0f;
    float max_range_x = 10.0f;
    float min_range_y = -10.0f;
    float max_range_y = 10.0f;
    float step = 0.1f;
    bool show_grid = true;
    bool show_data_points = true;
    int output_index = 0;
    // Dodane zmienne do zarządzania rozmiarem okna
    int windowWidth = 1400;
    int windowHeight = 960;

    float scale_factor = 1.0f;
};
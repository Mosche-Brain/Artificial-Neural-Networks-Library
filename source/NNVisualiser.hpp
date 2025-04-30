#pragma once
#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#include <thread>

#include "Network/NeuralNet.hpp"

class NNVisualiser
{
public:
    NNVisualiser(NeuralNet* network);
    ~NNVisualiser();

    void Render();
    void display();
    
    float inputs[2] = {0.0f, 0.0f};
    float training_epochs = 1000.0f;
    float learning_rate = 0.1f;
    std::string forward_result = "N/A";
    std::string training_status = "Untrained";

    MatrixXd train_x = (MatrixXd(4, 2) << 0.0, 0.0,
                                        0.0, 1.0,
                                        1.0, 0.0,
                                        1.0, 1.0).finished();
    MatrixXd train_y = (MatrixXd(4, 1) << 0.0,
                                        1.0,
                                        1.0,
                                        0.0).finished();


protected:
    void drawNeuron(glm::vec2 pos, float radius);
    void drawConnection(glm::vec2 from, glm::vec2 to, float weight);
    void drawCircle(float cx, float cy, float r, int num_segments);

    GLFWwindow* window;
    std::thread thread;
    NeuralNet* network;
};
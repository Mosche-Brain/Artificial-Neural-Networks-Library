#include "NNVisualiser.hpp"
#include <cmath>
#include <iostream>

NNVisualiser::NNVisualiser(NeuralNet* network)
{
    this->network = network;
}

NNVisualiser::~NNVisualiser()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void NNVisualiser::startTraining(int _iter_, float _rate_) 
{
    // Uruchamiamy trenowanie w osobnym wątku
    is_training = true;
    std::thread training_thread([=]() 
    {
        {
            // Blokujemy dostęp do sieci podczas aktualizacji
            std::lock_guard<std::mutex> lock(network_mutex);
            network->train(train_x, train_y, _iter_, _rate_); // Metoda trenowania (propagacja wsteczna)
            this->stopTraining();
        }
        // while (is_training) 
        // {
        //     // Przygotowanie danych treningowych (przykładowe)


        //     // Opcjonalne: krótka przerwa, aby nie obciążać procesora
        //     // std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // }
    });
    training_thread.detach(); // Odłączamy wątek (lub możemy go przechować do późniejszego dołączenia)
}

void NNVisualiser::stopTraining() 
{
    is_training = false;
}

void NNVisualiser::playGraph()
{
    static float min_range_x = -10.0f;
    static float max_range_x = 10.0f;
    static float min_range_y = -10.0f;
    static float max_range_y = 10.0f;
    static float step = 0.1f;
    static bool show_grid = true;
    static bool show_data_points = true;
    static int output_index = 0;

    int input_size = network->layers[0]->inputWidth();
    int output_size = network->layers.back()->size();
    
    ImGui::Begin("Graph Controls");
    ImGui::SliderFloat("Min X", &min_range_x, -50.0f, 0.0f);
    ImGui::SliderFloat("Max X", &max_range_x, 0.0f, 50.0f);
    ImGui::SliderFloat("Min Y", &min_range_y, -50.0f, 50.0f);
    ImGui::SliderFloat("Max Y", &max_range_y, -50.0f, 50.0f);
    ImGui::SliderFloat("Step", &step, 0.01f, 1.0f);
    ImGui::Checkbox("Show Grid", &show_grid);
    if (output_size > 1) 
    {
        ImGui::SliderInt("Output Index", &output_index, 0, output_size - 1);
    }
    ImGui::End();
    
    // ImGui::Begin("Network Control");

    // static int n_iter = 200;
    // static float rate = 0.1;

    // ImGui::SliderInt("Iter", &n_iter, 0, 2000);
    // ImGui::SliderFloat("rate", &rate, 0.0f, 1.0f);

    // if (ImGui::Button("Train") && !is_training) {
    //     startTraining(n_iter, rate);
    // }
 
    // ImGui::End();

    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(min_range_x, max_range_x, min_range_y, max_range_y, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    if (show_grid) 
    {
        glColor3f(0.2f, 0.2f, 0.2f);
        glLineWidth(1.0f);
        glBegin(GL_LINES);
        for (float x = std::ceil(min_range_x); x <= max_range_x; x += 1.0f) 
        {
            glVertex2f(x, min_range_y);
            glVertex2f(x, max_range_y);
        }
        for (float y = std::ceil(min_range_y); y <= max_range_y; y += 1.0f) 
        {
            glVertex2f(min_range_x, y);
            glVertex2f(max_range_x, y);
        }
        glEnd();
    }
    
    glColor3f(0.5f, 0.5f, 0.5f);
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glVertex2f(min_range_x, 0.0f);
    glVertex2f(max_range_x, 0.0f);
    glVertex2f(0.0f, min_range_y);
    glVertex2f(0.0f, max_range_y);
    glEnd();

    // Draw numeric labels on X and Y axes using ImGui
    // Get window size for projection
    int win_width, win_height;
    glfwGetFramebufferSize(window, &win_width, &win_height);

    // Project world coordinates to screen coordinates
    auto worldToScreen = [&](float x, float y) -> ImVec2 {
        float sx = (x - min_range_x) / (max_range_x - min_range_x) * win_width;
        float sy = win_height - (y - min_range_y) / (max_range_y - min_range_y) * win_height;
        return ImVec2(sx, sy);
    };

    // X axis labels
    for (float x = std::ceil(min_range_x); x <= max_range_x; x += 1.0f)
    {
        ImVec2 pos = worldToScreen(x, 0.0f);
        ImGui::GetBackgroundDrawList()->AddText(ImVec2(pos.x - 10, pos.y + 2), IM_COL32(200,200,200,255), std::to_string((int)x).c_str());
    }
    // Y axis labels
    for (float y = std::ceil(min_range_y); y <= max_range_y; y += 1.0f)
    {
        ImVec2 pos = worldToScreen(0.0f, y);
        ImGui::GetBackgroundDrawList()->AddText(ImVec2(pos.x + 4, pos.y - 7), IM_COL32(200,200,200,255), std::to_string((int)y).c_str());
    }
    
    if (input_size == 1) 
    {
        glColor3f(0.0f, 1.0f, 0.0f);
        glLineWidth(2.0f);
        glBegin(GL_LINE_STRIP);
        for (float x = min_range_x; x <= max_range_x; x += step) {
            MatrixXd input(1, 1);
            input << x;
            // VectorXd output = network->forward(input);
            VectorXd output = network->layers.back()->outputs;
            float y = output_size == 1 ? output[0] : output[output_index];
            glVertex2f(x, y);
        }
        glEnd();
    } 
    else if (input_size == 2) 
    {
        glPointSize(step * (max_range_x - min_range_x) * 10);
        glBegin(GL_POINTS);
        for (float x = min_range_x; x <= max_range_x; x += step) {
            for (float y = min_range_y; y <= max_range_y; y += step) {
                MatrixXd input(1, 2);
                input << x, y;
                VectorXd output = network->forward(input);
                float value = output_size == 1 ? output[0] : output[output_index];
                float normalized = std::tanh(value);
                float strength = std::abs(normalized);
                glColor3f(normalized > 0 ? 0.2f : strength, normalized > 0 ? strength : 0.2f, 0.2f);
                glVertex2f(x, y);
            }
        }
        glEnd();
    } 
    else 
    {
        ImGui::Begin("Warning");
        ImGui::Text("Visualization supports only 1D or 2D inputs.");
        ImGui::End();
    }
}

void NNVisualiser::display()
{
    if (!glfwInit())
    {
        std::cerr << "Failed to init GLFW\n";
        return;
    }

    int windowWidth = 1400;
    int windowHeight = 960;

    window = glfwCreateWindow(windowWidth, windowHeight, "Neural Graph", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, GL_TRUE);
    ImGui_ImplOpenGL3_Init();

    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to init GLEW\n";
        return;
    }

    // Add window size callback
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
    });

    uint view_mode = 0    ;

    while (!glfwWindowShouldClose(window))
    {
        glClearColor(0.f, 0.f, 0.f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Get current window size for proper rendering
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        
        ImGui::Begin("Control Panel");
        static int n_iter = 200;
        static float rate = 0.1;
        static VectorXd input(network->input.size());

        ImGui::Text("Neural Network Control");
        
        ImGui::SliderInt("Iter", &n_iter, 0, 2000);
        ImGui::SliderFloat("rate", &rate, 0.0f, 1.0f);
         
        if (ImGui::Button("Train") && !is_training) 
        {
            startTraining(n_iter, rate);
        }


        float inputs[input.size()];
        for(int i = 0 ; i < input.size() ; i++)
        {
            std::string label = "input " + i;
            ImGui::InputFloat(label.c_str(), &inputs[0], 0.0f, 1.0f, "%.1f");
        }

        // ImGui::InputFloat("Input 2", &inputs[1], 0.0f, 1.0f, "%.1f");
        
        if (ImGui::Button("Run Forward")) 
        {
            VectorXd network_output = network->forward(input);
            forward_result = std::to_string(network_output[0]);
        }
        ImGui::Text("Forward Result: %s", forward_result.c_str());
        ImGui::End();
    

        switch (view_mode)
        {
            case 0:
            {
                // Update projection matrix for neural network visualization
                glMatrixMode(GL_PROJECTION);
                glLoadIdentity();
                glOrtho(0, width, height, 0, -1, 1);
                glMatrixMode(GL_MODELVIEW);
                glLoadIdentity();

                Render();
            
    
                if(ImGui::TreeNode("Layers"))
                {
                    for(int i = 0 ; i < this->network->layers.size() ; i++)
                    {
                        std::string label = "Layer " + std::to_string(i);
                        if(ImGui::TreeNode(label.c_str()))
                        {
                            ImGui::Text("size: %d", this->network->layers[i]->size());
                            
                            if(ImGui::TreeNode("Weights"))
                            {
                                for(int j = 0 ; j < network->layers[i]->size() ; j++)
                                {
                                    std::string weights_label = "Row " + std::to_string(j);
                                    if(ImGui::TreeNode(weights_label.c_str()))
                                    {
                                        for(int k = 0 ; k < network->layers[i]->weights.row(j).size() ; k++)
                                        {
                                            double weight = network->layers[i]->weights.row(j)[k];
                                            ImGui::Text("%.8f", weight);
                                        }
    
                                        ImGui::TreePop();
                                    }
                                }
    
                                ImGui::TreePop();
                            }
    
                            if(ImGui::TreeNode("output"))
                            {
                                for(int j = 0 ; j < this->network->layers[i]->size() ; j++)
                                {
                                    ImGui::Text("%.8f", network->layers[i]->outputs[j]);
                                }
    
                                ImGui::TreePop();
                            }
    
                            ImGui::TreePop();
                        }
                    }
    
                    ImGui::TreePop();
                }
                break;
            } 
        
            case 1:
            {
                // Update projection matrix for graph visualization
                glMatrixMode(GL_PROJECTION);
                glLoadIdentity();
                glOrtho(min_range_x, max_range_x, min_range_y, max_range_y, -1, 1);
                glMatrixMode(GL_MODELVIEW);
                glLoadIdentity();
                
                playGraph();
            }
            default: break;
        }
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
}

void NNVisualiser::Render()
{
    // Get current window size
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    
    float x_spacing = width * 0.1f;  // Scale with window width (10% of width)
    float y_spacing = height * 0.06f; // Scale with window height (6% of height)
    float radius = std::min(width, height) * 0.02f; // Scale radius (2% of min dimension)

    std::vector<std::vector<glm::vec2>> positions;
    for (size_t i = 0; i < network->layers.size(); ++i)
    {
        Layer* layer = network->layers[i];
        std::vector<glm::vec2> layer_positions;

        float total_height = layer->size() * (radius * 2 + y_spacing);
        float y_offset = (height - total_height) / 2;

        for (int j = 0; j < layer->size(); ++j)
        {
            glm::vec2 pos = 
            {
                width * 0.1f + i * x_spacing,
                y_offset + j * (radius * 2 + y_spacing)
            };
            
            float normalized = std::tanh(layer->outputs[j]);
            float strength = std::abs(normalized);
                        
            glm::vec3 color = normalized > 0 ? glm::vec3(0.2f, strength, 0.2f) : glm::vec3(strength, 0.2f, 0.2f);

            glColor3f(color.x / 4, color.y / 4, color.z / 4);
            drawNeuron(pos, radius);

            glColor3f(color.x, color.y, color.z);
            drawCircle(pos.x, pos.y, radius, 20);
            
            layer_positions.push_back(pos);
        }
        
        positions.push_back(layer_positions);
    }
    
    for (size_t i = 0; i < positions.size() - 1; ++i)
    {
        Layer* from_layer = network->layers[i];
        Layer* to_layer   = network->layers[i + 1];

        for (int from_idx = 0; from_idx < from_layer->size(); ++from_idx)
        {
            for (int to_idx = 0; to_idx < to_layer->size(); ++to_idx)
            {
                float weight = to_layer->weights(to_idx, from_idx);
                drawConnection(
                    positions[i][from_idx],
                    positions[i + 1][to_idx],
                    weight
                );
            }
        }
    }
}

void NNVisualiser::drawNeuron(glm::vec2 pos, float radius)
{
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(pos.x, pos.y);
    for (int i = 0; i <= 20; ++i)
    {
        float angle = i * 2.0f * M_PI / 20;
        float x = pos.x + std::cos(angle) * radius;
        float y = pos.y + std::sin(angle) * radius;
        glVertex2f(x, y);
    }
    glEnd();
}

void NNVisualiser::drawConnection(glm::vec2 from, glm::vec2 to, float weight)
{
    float normalized = std::tanh(weight);
    float strength = std::abs(normalized);

    if (normalized > 0)
        glColor3f(0.2f, strength, 0.2f);
    else
        glColor3f(strength, 0.2f, 0.2f);

    glLineWidth(1.0f + 2.0f * strength);

    glBegin(GL_LINES);
    glVertex2f(from.x, from.y);
    glVertex2f(to.x, to.y);
    glEnd();
}

void NNVisualiser::drawCircle(float cx, float cy, float r, int num_segments)
{
    float theta = 3.14 * 2 / float(num_segments);
    float tangetial_factor = tanf(theta);
    float radial_factor = cosf(theta);
    float x = r;
    float y = 0;
    glLineWidth(2);
    glBegin(GL_LINE_LOOP);
    for (int ii = 0; ii < num_segments; ii++)
    {
        glVertex2f(x + cx, y + cy);
        float tx = -y;
        float ty = x;
        x += tx * tangetial_factor;
        y += ty * tangetial_factor;
        x *= radial_factor;
        y *= radial_factor;
    }
    glEnd();
} 
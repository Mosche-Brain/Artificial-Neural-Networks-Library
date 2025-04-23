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

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, windowWidth, windowHeight, 0, -1, 1);  // 2D

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    while (!glfwWindowShouldClose(window))
    {
        //glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClearColor(0.f, 0.f, 0.f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        Render();
    
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Info");
        ImGui::Text("Neural Network");
        ImGui::End();

        if(ImGui::TreeNode("Layers"))
        {
            for(int i = 0 ; i < this->network->layers.size() ; i++)
            {
                std::string label = "Layer " + std::to_string(i);
                if(ImGui::TreeNode(label.c_str()))
                {
                    // std::string size_label = "size: " + std::to_string(this->network->layers[i]->layer_size);
                    // ImGui::Text(size_label.c_str());
                    ImGui::Text("size: %d", this->network->layers[i]->layer_size);
                    
                    if(ImGui::TreeNode("Weights"))
                    {
                        for(int j = 0 ; j < network->layers[i]->layer_size ; j++)
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
                        for(int j = 0 ; j < this->network->layers[i]->layer_size ; j++)
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

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
}

void NNVisualiser::Render()
{
    float x_spacing = 150.0f;
    float y_spacing = 60.0f;
    float radius = 20.0f;

    std::vector<std::vector<glm::vec2>> positions;
    for (size_t i = 0; i < network->layers.size(); ++i)
    {
        Layer* layer = network->layers[i];
        std::vector<glm::vec2> layer_positions;

        float total_height = layer->layer_size * (radius * 2 + y_spacing);
        float y_offset = (600 - total_height) / 2;

        for (int j = 0; j < layer->layer_size; ++j)
        {
            glm::vec2 pos = {
                100.0f + i * x_spacing,
                y_offset + j * (radius * 2 + y_spacing)
            };
            
            
            // float minOutputValue = -1;
            // float maxOutputValue = -1;
            
            // float x = std::max((float)minOutputValue, std::min((float)layer->outputs[j],  maxOutputValue));
            
            // float t, r, g, b;
            
            // if(x <= 0)
            // {
            //     t = (x + 1) / 1;      
            //     r = t;                
            //     g = 1.0;              
            // }
            // else
            // {
            //     t = x / 1;            
            //     r = 1.0;
            //     g = 1.0 - t;
            // }
            
            // glColor4f(r/4, g/4, 0, 0.5f);


            float normalized = std::tanh(layer->outputs[j]); // normalize weights to [-1,1]
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
    
    
    // Draw connections
    for (size_t i = 0; i < positions.size() - 1; ++i)
    {
        Layer* from_layer = network->layers[i];
        Layer* to_layer   = network->layers[i + 1];

        for (int from_idx = 0; from_idx < from_layer->layer_size; ++from_idx)
        {
            for (int to_idx = 0; to_idx < to_layer->layer_size; ++to_idx)
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
    //glColor3f(0.8f, 0.8f, 0.9f);
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
    float normalized = std::tanh(weight); // normalize weights to [-1,1]
    float strength = std::abs(normalized);

    if (normalized > 0)
        glColor3f(0.2f, strength, 0.2f); // green
    else
        glColor3f(strength, 0.2f, 0.2f); // red

    glLineWidth(1.0f + 2.0f * strength);

    glBegin(GL_LINES);
    glVertex2f(from.x, from.y);
    glVertex2f(to.x, to.y);
    glEnd();
}




void NNVisualiser::drawCircle(float cx, float cy, float r, int num_segments)
{
    float theta = 3.1415926 * 2 / float(num_segments);
    float tangetial_factor = tanf(theta);//calculate the tangential factor 

    float radial_factor = cosf(theta);//calculate the radial factor 

    float x = r;//we start at angle = 0 

    float y = 0;
    glLineWidth(2);
    glBegin(GL_LINE_LOOP);
    for (int ii = 0; ii < num_segments; ii++)
    {
        glVertex2f(x + cx, y + cy);//output vertex 

        //calculate the tangential vector 
        //remember, the radial vector is (x, y) 
        //to get the tangential vector we flip those coordinates and negate one of them 

        float tx = -y;
        float ty = x;

        //add the tangential vector 

        x += tx * tangetial_factor;
        y += ty * tangetial_factor;

        //correct using the radial factor 

        x *= radial_factor;
        y *= radial_factor;
    }
    glEnd();
}
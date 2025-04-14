#pragma once
#include <SFML/Graphics.hpp>
#include <thread>

#include "Network/NeuralNet.hpp"

typedef struct vec2
{
    float x, y;
};

class NNVisualiser
{
public:
    NNVisualiser(NeuralNet* network);
    ~NNVisualiser();

    void Render();
    void drawNeuron(VirtualNeuron neuron, vec2 position);
    void drawLayer(int index);

protected:
    std::thread thread;
    NeuralNet* network;
};
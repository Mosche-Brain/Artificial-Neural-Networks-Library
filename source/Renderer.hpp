#pragma once
#include "Network/NeuralNet.hpp"

class NNVisualiser
{
public:
    NNVisualiser(NeuralNet* network);

    void Render();
    void drawNeuron();

protected:
    NeuralNet* network;
};
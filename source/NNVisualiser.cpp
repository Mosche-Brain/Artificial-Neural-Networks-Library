#include "NNVisualiser.hpp"

NNVisualiser::NNVisualiser(NeuralNet* network)
{
    this->network = network;
}

void NNVisualiser::Render()
{
    float spacing = 10.0f;


    for(int i = 0 ; i < network->layers.size() ; i++)
    {
        Layer* layer = network->layers[i];

        sf::Vector2 position(0,0);
        for(int j = 0 ; j < layer->layer_size ; j++)
        {

            sf::CircleShape
        }
    }
}

void NNVisualiser::drawLayer(int index)
{
    // Layer* layer = this->network.layers(index);


    // for(int i = 0 ; i < layer->layer_size ; i++)
    // {
    //     sf::CircleShape
    // }
}
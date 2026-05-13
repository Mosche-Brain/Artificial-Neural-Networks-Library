#include "LayerBase.hpp"

namespace YANN::Models::Layers
{
    void LayerBase::initParameters(int layerSize, int inputLenght)
    {
        // this->weights = cum::Matrix::Random(layerSize, inputLenght) * static_cast<cum::cumeric_t>(0.01f);
        // this->biases  = cum::Matrix::Zero(layerSize, 1)             ;
        // this->biases  = vector_t::Random(layerSize, 1) * static_cast<cum::cumeric_t>(0.1f);
        // this->outputs = cum::Matrix::Zero(layerSize, 1);
        // this->inputs  = cum::Matrix::Zero(layerSize, inputLenght);

        // this->weights = YANN::math_api::createRandomMatrix(layerSize, inputLenght, static_cast<cum::cumeric_t>(-0.01f), static_cast<cum::cumeric_t>(0.01f));
        // this->biases  = YANN::math_api::createMatrix(layerSize, 1, static_cast<cum::cumeric_t>(0));
        // this->outputs = YANN::math_api::createMatrix(layerSize, 1, static_cast<cum::cumeric_t>(0));
        // this->inputs  = YANN::math_api::createMatrix(layerSize, inputLenght, static_cast<cum::cumeric_t>(0));

        this->weights = cum::Matrix(layerSize, inputLenght, 0.001);
        this->biases  = cum::Vector(layerSize, 0);
        this->outputs = cum::Matrix(layerSize, 1, 0);
        this->inputs  = cum::Matrix(layerSize, inputLenght, 0);

        this->_initialized_ = true;
    }

    bool LayerBase::initialized()
    {
        return this->_initialized_;
    }

    int LayerBase::size()
    {
        return this->_layerSize_;
    }

    cum::Matrix LayerBase::Outputs()
    {
        return this->outputs;
    }
    
    cum::Matrix LayerBase::Inputs()
    {
        return this->inputs;
    }
    
    cum::Matrix LayerBase::Weights()
    {
        return this->weights;
    }

    cum::Vector LayerBase::Biases()
    {
        return this->biases;
    }

    LAYER_TYPE LayerBase::layerType()
    {
        return _layerType_;
    }
}
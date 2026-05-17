#include "LayerBase.hpp"

namespace YANN::Models::Layers
{
    void LayerBase::initParameters(int layerSize, int inputLenght)
    {
        this->weights = cum::Matrix(layerSize, inputLenght, 0.001);
        this->biases  = cum::Matrix(layerSize, 1, 0);
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

    cum::Matrix& LayerBase::Outputs()
    {
        return this->outputs;
    }

    cum::Matrix& LayerBase::Inputs()
    {
        return this->inputs;
    }
    
    cum::Matrix& LayerBase::Weights()
    {
        return this->weights;
    }

    cum::Matrix& LayerBase::Biases()
    {
        return this->biases;
    }

    LAYER_TYPE LayerBase::layerType()
    {
        return _layerType_;
    }
}
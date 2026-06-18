#include "LayerBase.hpp"

namespace YANN::Models::Layers
{
    void LayerBase::initParameters(int layerSize, int inputLenght)
    {
        this->weights = cum::Matrix::Random(layerSize, inputLenght) * 0.1_c;
        this->biases  = cum::Matrix::Zeros(layerSize, 1);
        this->outputs = cum::Matrix::Zeros(layerSize, 1);
        this->inputs  = cum::Matrix::Zeros(layerSize, inputLenght);
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
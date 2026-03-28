#include "LayerBase.hpp"

namespace YANN::Models::Layers
{
    void LayerBase::initParameters(int layerSize, int inputLenght)
    {
        this->weights = matrix_t::Random(layerSize, inputLenght) * static_cast<numeric_t>(0.01f);
        // this->biases  = matrix_t::Zero(layerSize, 1)             ;
        this->biases  = vector_t::Random(layerSize, 1) * static_cast<numeric_t>(0.1f);
        this->outputs = matrix_t::Zero(layerSize, 1);
        this->inputs  = matrix_t::Zero(layerSize, inputLenght);

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

    matrix_t LayerBase::Outputs()
    {
        return this->outputs;
    }
    
    matrix_t LayerBase::Inputs()
    {
        return this->inputs;
    }
    
    matrix_t LayerBase::Weights()
    {
        return this->weights;
    }

    matrix_t LayerBase::Biases()
    {
        return this->biases;
    }

    LAYER_TYPE LayerBase::layerType()
    {
        return _layerType_;
    }
}
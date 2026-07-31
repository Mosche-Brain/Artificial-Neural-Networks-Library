#include "LayerBase.hpp"

namespace yann::models::layers
{
    void LayerBase::initParameters(int layerSize, int inputLenght)
    {
        this->weights       = Parameter::Uniform(layerSize, inputLenght);   /* neurons * input_length */
        this->biases        = Parameter::Zeros(layerSize, 1);               /* Column-Vector */
        this->outputs       = cum::Matrix::Zeros(layerSize, 1);             /* Column-Vector */
        this->raw_outputs   = cum::Matrix::Zeros(layerSize, 1);             /* Column-Vector */
        this->inputs        = cum::Matrix::Zeros(layerSize, inputLenght);   /* Column-Vector */
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
        return this->weights();
    }

    cum::Matrix& LayerBase::Biases()
    {
        return this->biases();
    }

    LAYER_TYPE LayerBase::layerType()
    {
        return _layerType_;
    }
}
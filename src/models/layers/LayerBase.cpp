#include "LayerBase.hpp"

namespace yann::models::layers
{
    // void LayerBase::initParameters(int layerSize, int inputLenght)
    // {
        // this->weights       = Parameter::Uniform(layerSize, inputLenght);   /* neurons * input_length */
        // this->biases        = Parameter::Zeros(layerSize, 1);               /* Column-Vector */
        // this->cache.a       = cum::Matrix::Zeros(layerSize, 1);             /* Column-Vector */
        // this->cache.z   	= cum::Matrix::Zeros(layerSize, 1);             /* Column-Vector */
        // this->cache.x       = cum::Matrix::Zeros(inputLenght, 1);   /* Column-Vector */
        // // this->cache.x        = cum::Matrix::Zeros(layerSize, inputLenght);   /* Column-Vector */
        // this->_initialized_ = true;
    // }


    bool LayerBase::initialized()
    {
        return this->_initialized_;
    }

    int LayerBase::size()
    {
        return this->_layerSize_;
    }

    cum::Tensor& LayerBase::Outputs()
    {
        // return this->outputs;
        return this->cache.a;
    }

    cum::Tensor& LayerBase::Inputs()
    {
        // return this->inputs;
        return this->cache.x;
    }
    
    cum::Tensor& LayerBase::weights()
    {
        // return this->weights();
    }

    cum::Tensor& LayerBase::weights_grad()
    {
        // return this->weights.gradient;
    }

    cum::Tensor& LayerBase::biases()
    {
        // return this->biases();
    }

    cum::Tensor& LayerBase::biases_grad()
    {
        // return this->biases.gradient;
    }

    LAYER_TYPE LayerBase::layerType()
    {
        return _layerType_;
    }
}

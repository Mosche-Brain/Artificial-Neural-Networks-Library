#include "LayerBase.hpp"

namespace yann::models::layers
{
    void LayerBase::initParameters(int layerSize, int inputLenght)
    {
        this->weights = Parameter::Uniform(layerSize, inputLenght);
        this->biases  = Parameter::Zeros(layerSize, 1); /* Column-Vector */
        this->outputs = cum::Matrix::Zeros(layerSize, 1); /* Column-Vector */
        this->inputs  = cum::Matrix::Zeros(layerSize, inputLenght); /* Column-Vector */
        this->_initialized_ = true;
        // this->weights = cum::Matrix::Random(layerSize, inputLenght) * 0.1_c;
        // this->biases  = cum::Matrix::Zeros(layerSize, 1); /* Column-Vector */
        // this->outputs = cum::Matrix::Zeros(layerSize, 1); /* Column-Vector */
        // this->inputs  = cum::Matrix::Zeros(layerSize, inputLenght); /* Column-Vector */
        // this->_initialized_ = true;
    }

    // std::vector<Parameter*> LayerBase::collect_params()
    // {
    //     std::vector<Parameter*> params;
    //     params.push_back(&weights);
    //     params.push_back(&biases);
    //     return params;
    // }

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
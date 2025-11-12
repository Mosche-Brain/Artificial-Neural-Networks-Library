#include "LayerBase.hpp"

namespace ANN::Models::Layers
{
    void LayerBase::initParameters(int layerSize, int inputLenght)
    {
        this->weights = Eigen::MatrixXf::Random(layerSize, inputLenght) * 0.01f;
        // this->biases  = Eigen::MatrixXf::Zero(layerSize, 1)             ;
        this->biases  = Eigen::MatrixXf::Random(layerSize, 1) * 0.1f;
        this->outputs = Eigen::MatrixXf::Zero(layerSize, 1);
        this->inputs  = Eigen::MatrixXf::Zero(layerSize, inputLenght);

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

    Eigen::MatrixXf LayerBase::Outputs()
    {
        return this->outputs;
    }
    
    Eigen::MatrixXf LayerBase::Inputs()
    {
        return this->inputs;
    }
    
    Eigen::MatrixXf LayerBase::Weights()
    {
        return this->weights;
    }

    Eigen::MatrixXf LayerBase::Biases()
    {
        return this->biases;
    }

    LAYER_TYPE LayerBase::layerType()
    {
        return _layerType_;
    }
}
#include "LayerBase.hpp"

namespace SNN::Models::Layers
{
    void LayerBase::initParameters(int layerSize, int inputLenght)
    {
        this->weights = Eigen::MatrixXf::Random(layerSize, inputLenght) * 0.1f;
        this->biases  = Eigen::MatrixXf::Random(layerSize, 1)           * 0.1f;
        this->outputs = Eigen::MatrixXf::Random(layerSize, 1);

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
    
    Eigen::MatrixXf LayerBase::Weights()
    {
        return this->weights;
    }

    Eigen::MatrixXf LayerBase::Biases()
    {
        return this->biases;
    }
}
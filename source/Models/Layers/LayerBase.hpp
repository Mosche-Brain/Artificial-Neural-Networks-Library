#pragma once

#include <Eigen/Dense>

#include <memory>

#include "Utility/Activation.hpp"

namespace SNN::Models::Layers
{
    class LayerBase
    {
    public:
        virtual ~LayerBase() = default;

        virtual void initParameters(int layerSize, int inputLenght);
        
        virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input) = 0;
        virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& deltaOutput) = 0;
        virtual void update_weights(float_t rate) = 0;
        
        // virtual std::unique_ptr<LayerBase> getUnique() = 0;
        
        bool initialized();
        virtual int size();
        virtual Eigen::MatrixXf Outputs();
        virtual Eigen::MatrixXf Weights();
        virtual Eigen::MatrixXf Biases();
        Utils::Activation activation;
        
    protected:
        Eigen::MatrixXf weights;
        Eigen::VectorXf biases;
        Eigen::MatrixXf outputs;

        int _layerSize_;
        bool _initialized_ = false;
    };
}   
#pragma once

#include <memory>
#include <vector>

#include <cum/Matrix.hpp>
#include <cum/Vector.hpp>
#include <cum/LinearAlgebra.hpp>
#include <cum/functions.hpp>

#include "LayerType.hpp"
#include "YANN/Parameter.hpp"

namespace yann::models::layers
{
    class LayerBase
    {
    public:


        virtual ~LayerBase() = default;

        virtual void initParameters(int layerSize, int inputLenght);
        
        // virtual cum::Vector forward(const cum::Vector& input) = 0;
        virtual cum::Matrix forward(const cum::Matrix& input) = 0;
        virtual cum::Matrix backward(const cum::Matrix& deltaOutput) = 0;
        virtual void update_weights(const cum::cumeric_t rate) = 0;

        // std::vector<Parameter*> collect_params();

        virtual void collect_parameters(std::vector<Parameter*>& params) = 0;
        // virtual std::unique_ptr<LayerBase> getUnique() = 0;
        
        bool initialized();
        virtual int size();
        virtual cum::Matrix& Outputs();
        virtual cum::Matrix& Inputs();
        virtual cum::Matrix& Weights();
        virtual cum::Matrix& Biases();
        // utils::Activation activation;
        cum::functions::activation_t activation;
        
        LAYER_TYPE layerType();
    protected:
        Parameter weights;
        Parameter biases;
        // cum::Matrix weights;
        // cum::Matrix biases;
        cum::Matrix outputs;
        cum::Matrix preactivatedOutputs;
        cum::Matrix inputs;
           
        int _layerSize_;
        bool _initialized_ = false;
        LAYER_TYPE _layerType_;
    };
}   
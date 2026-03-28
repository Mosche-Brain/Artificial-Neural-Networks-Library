#pragma once

#include "build_config.hpp"
#include <Eigen/Dense>
#include <memory>

#include "Utility/Activation.hpp"
#include "Models/Layers/LayerType.hpp"

namespace YANN::Models::Layers
{
    class LayerBase
    {
    public:
        virtual ~LayerBase() = default;

        virtual void initParameters(int layerSize, int inputLenght);
        
        virtual matrix_t forward(const matrix_t& input) = 0;
        virtual matrix_t backward(const matrix_t& deltaOutput) = 0;
        virtual void update_weights(numeric_t rate) = 0;
        
        // virtual std::unique_ptr<LayerBase> getUnique() = 0;
        
        bool initialized();
        virtual int size();
        virtual matrix_t Outputs();
        virtual matrix_t Inputs();
        virtual matrix_t Weights();
        virtual matrix_t Biases();
        Utils::Activation activation;
        
        LAYER_TYPE layerType();
    protected:
        matrix_t weights;
        vector_t biases;
        matrix_t outputs;
        matrix_t preactivatedOutputs;
        matrix_t inputs;
           
        int _layerSize_;
        bool _initialized_ = false;
        LAYER_TYPE _layerType_;
    };
}   
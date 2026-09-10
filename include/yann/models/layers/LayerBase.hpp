#pragma once

#include <memory>
#include <vector>

#include <cum/Matrix.hpp>
#include <cum/Vector.hpp>
#include <cum/LinearAlgebra.hpp>
#include <cum/functions.hpp>

#include "LayerType.hpp"
#include "cum/Tensor.hpp"
#include "yann/Parameter.hpp"

namespace yann::models::layers
{
    struct ForwardCache
    {
        cum::Matrix x;
        cum::Matrix z;
        cum::Matrix a;
        cum::Matrix dz;
        cum::Matrix da;

        void clear()
        {
            x.fill(0);
            z.fill(0);
            a.fill(0);
            dz.fill(0);
        }

        void resize(cum::dim_t in_features, cum::dim_t out_features, cum::dim_t batch_size)
        {
            x = cum::Matrix(in_features, batch_size);
            z = cum::Matrix(out_features, batch_size);
            a = cum::Matrix(out_features, batch_size);
            dz= cum::Matrix(out_features, batch_size);
        }
    };

    class LayerBase
    {
    public:
        virtual ~LayerBase() = default;

        virtual void initParameters(int output_features, int input_features) = 0;
        // dofdam też inicjalizator parametrów przyjmujący referencje do poprzedniej warstwy
        
        // virtual cum::Vector forward(const cum::Vector& input) = 0;
        virtual cum::Tensor forward(const cum::Tensor& input) {};
        virtual cum::Tensor backward(const cum::Tensor& input) {};

        virtual cum::Matrix forward(const cum::Matrix& input) = 0;
        virtual cum::Matrix backward(const cum::Matrix& deltaOutput) = 0;

        virtual void collect_parameters(std::vector<Parameter*>& params) = 0;
        // virtual std::unique_ptr<LayerBase> getUnique() = 0;
        
        bool initialized();
        virtual int size();
        virtual cum::Matrix& Outputs();
        virtual cum::Matrix& Inputs();
        virtual cum::Matrix& weights_grad();
        virtual cum::Matrix& weights();
        virtual cum::Matrix& biases();
        virtual cum::Matrix& biases_grad();
        // utils::Activation activation;
        cum::functions::activation_t activation;
        
        LAYER_TYPE layerType();
    protected:
        ForwardCache cache;
        // cum::Matrix outputs;
        // cum::Matrix raw_outputs;
        // cum::Matrix inputs;
           
        int _layerSize_;
        bool _initialized_ = false;
        LAYER_TYPE _layerType_;
    };
}   
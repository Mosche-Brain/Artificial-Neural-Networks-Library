#pragma once

#include <memory>
#include <vector>

#include <cum/Matrix.hpp>
#include <cum/Vector.hpp>
#include <cum/LinearAlgebra.hpp>
#include <cum/functions.hpp>
#include <cum/Tensor.hpp>

#include "yann/models/layers/LayerType.hpp"
#include "yann/Parameter.hpp"

namespace yann::models::layers
{
    // TODO: Do not implicitly reserve memory necessary for training (da, dz, etc)
    struct ForwardCache // ForwardCache name may not be relevant due to storing dz and da, more general name like a `LayerCache` may be better
    {
        cum::Tensor x;
        cum::Tensor z;
        cum::Tensor a;
        cum::Tensor dz;
        cum::Tensor da;

        void clear()
        {
            x.fill(0);
            z.fill(0);
            a.fill(0);
            dz.fill(0);
        }

        void resize(cum::dim_t in_features, cum::dim_t out_features, cum::dim_t batch_size)
        {
            x   = cum::Tensor::make_matrix(in_features, batch_size);
            z   = cum::Tensor::make_matrix(out_features, batch_size);
            a   = cum::Tensor::make_matrix(out_features, batch_size);
            dz  = cum::Tensor::make_matrix(out_features, batch_size);
        }

        void resise(const cum::Shape& input_shape, const cum::Shape& output_shape, cum::dim_t batch_size)
        {
            cum::Shape in_shape = input_shape;
            cum::Shape out_shape = input_shape;

            if(batch_size > 1)
            {
                in_shape.insert(input_shape.begin(), batch_size);
                out_shape.insert(output_shape.begin(), batch_size);
            }

            x   = cum::Tensor(in_shape);
            z   = cum::Tensor(out_shape);
            a   = cum::Tensor(out_shape);
            dz  = cum::Tensor(out_shape);
            da  = cum::Tensor(out_shape);
        }
    };

    class LayerBase
    {
    public:
        virtual ~LayerBase() = default;

        virtual void init_parameters(int output_features, int input_features) = 0; // I will swap output and input festures order
        virtual void init_parameters(const cum::Shape& input_shape, const cum::Shape& output_shape) {};
        // dofdam też inicjalizator parametrów przyjmujący referencje do poprzedniej warstwy
        
        virtual cum::Tensor forward(const cum::Tensor& input) = 0;
        virtual cum::Tensor backward(const cum::Tensor& input) = 0;

        virtual void collect_parameters(std::vector<Parameter*>& params) = 0;

        bool initialized();
        virtual int size();

        virtual cum::Tensor& Outputs();
        virtual cum::Tensor& Inputs();
        virtual cum::Tensor& weights_grad();
        virtual cum::Tensor& weights();
        virtual cum::Tensor& biases();
        virtual cum::Tensor& biases_grad();
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
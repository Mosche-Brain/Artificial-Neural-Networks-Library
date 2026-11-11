#include <stdexcept>
#include "LayerBase.hpp"

namespace yann::models::layers
{
    void LayerBase::init_parameters(int output_features, int input_features)
    {
        init_parameters(cum::Shape{static_cast<cum::dim_t>(input_features)},
                        cum::Shape{static_cast<cum::dim_t>(output_features)});
    }

    void LayerBase::init_parameters(const cum::Shape& input_shape, const cum::Shape& output_shape)
    {
        this->input_shape_ = input_shape;
        this->output_shape_ = output_shape;
        this->_initialized = true;
    }

    bool LayerBase::initialized()
    {
        return this->_initialized;
    }

    const cum::Shape& LayerBase::input_shape() const
    {
        return input_shape_;
    }

    const cum::Shape& LayerBase::output_shape() const
    {
        return output_shape_;
    }

    int LayerBase::size()
    {
        if (output_shape_.empty())
            return 0;
        return static_cast<int>(output_shape_.back());
    }

    cum::Tensor& LayerBase::Outputs()
    {
        // return this->cache.a;
    }

    cum::Tensor& LayerBase::Inputs()
    {
        // return this->cache.x;
    }
    
    cum::Tensor& LayerBase::weights()
    {
        throw std::logic_error("Layer does not have trainable weights");
    }

    cum::Tensor& LayerBase::weights_grad()
    {
        throw std::logic_error("Layer does not have trainable weight gradients");
    }

    cum::Tensor& LayerBase::biases()
    {
        throw std::logic_error("Layer does not have trainable biases");
    }

    cum::Tensor& LayerBase::biases_grad()
    {
        throw std::logic_error("Layer does not have trainable bias gradients");
    }

    Kind LayerBase::layerType()
    {
        return _kind;
    }
}

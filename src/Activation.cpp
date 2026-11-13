/*
 * @author: jaro
 * @name:   Activation
 * @file:   src/Activation.cpp
 * @date:   10 November 2026 22:07:04
 */

#include "yann/Activaton.hpp"

namespace cum::dnn { using namespace cum::neural_primitives; }

namespace yann
{
    Activation::Activation(const std::string& label)
    {
        *this = from_name(label);
    }

    Activation::Activation(const char* label)
    {
        *this = from_name(label ? label : "");
    }

    Activation::Activation(std::string_view label)
    {
        *this = from_name(label);
    }

    Activation::Activation(Tag)
    {
        *this = from_tag(tag);
    }

    Activation::Activation(cum::neural_primitives::Algo fwd, cum::neural_primitives::Algo bwd, Tag tag, const std::string& label)
        : fwd(fwd), bwd(bwd), tag(tag), label(label)
    { }

    // Activation Activation::from_name(const std::string& label)
    Activation Activation::from_name(std::string_view label)
    {
        if (label == "relu") return relu();
        if (label == "gelu") return gelu();
        if (label == "elu") return elu();
        if (label == "sigmoid") return sigmoid();
        if (label == "tanh") return tanh();
        throw std::runtime_error("Activation::from_name unknown activation label");
    }

    Activation Activation::from_tag(Tag tag)
    {
        switch (tag)
        {
            case Tag::relu: return relu();
            case Tag::gelu: return gelu();
            case Tag::elu: return elu();
            case Tag::tanh: return tanh();
            case Tag::sigmoid: return sigmoid();
            default: throw std::runtime_error("Activation::from_tag unknown activation tag");
        }
    }

    Activation Activation::relu()
    {
        return { cum::dnn::Algo::eltwise_relu, cum::dnn::Algo::eltwise_relu, Tag::relu, "relu" };
    }

    Activation Activation::elu()
    {
        return { cum::dnn::Algo::eltwise_elu, cum::dnn::Algo::eltwise_elu, Tag::elu, "elu" };
    }

    Activation Activation::gelu()
    {
        return { cum::dnn::Algo::eltwise_gelu, cum::dnn::Algo::eltwise_gelu, Tag::gelu, "gelu" };
    }

    Activation Activation::tanh()
    {
        return { cum::dnn::Algo::eltwise_tanh, cum::dnn::Algo::eltwise_tanh, Tag::tanh, "tanh" };
    }

    Activation Activation::sigmoid()
    {
        return { cum::dnn::Algo::eltwise_sigmoid, cum::dnn::Algo::eltwise_sigmoid, Tag::sigmoid, "sigmoid" };
    }


}

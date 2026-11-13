/*
 * @author: jaro
 * @name:   Activaton
 * @file:   include/yann/Activaton.hpp
 * @date:   10 November 2026 21:15:22
 */

#pragma once

#include <cum/neural_primitives/PrimitiveDescriptor.hpp>
#include <cum/neural_primitives/Primitive.hpp>

namespace yann
{
    struct Activation
    {
        enum class Tag { relu, gelu, elu, sigmoid, tanh, undef };

        Activation(const std::string&);
        Activation(const char*);
        Activation(std::string_view);
        Activation(Tag);
        Activation(cum::neural_primitives::Algo fwd, cum::neural_primitives::Algo bwd, Tag tag, const std::string& label);

        // static Activation from_name(const std::string& label);
        static Activation from_name(std::string_view label);
        static Activation from_tag(Tag tag);
        static Activation relu();
        static Activation gelu();
        static Activation elu();
        static Activation sigmoid();
        static Activation tanh();

        cum::neural_primitives::Algo fwd;
        cum::neural_primitives::Algo bwd;
        Tag tag = Tag::undef;
        std::string label;
    };
}
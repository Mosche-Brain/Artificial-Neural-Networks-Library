//
// Created by jaro on 7/18/26.
//


#include <cum/cum.hpp>
#include <cum/random.hpp>
#include <cum/memory.hpp>

#include <cmath>
#include <print>
#include <stdexcept>
#include <string>

#include "yann/Parameter.hpp"

#include "cum/runtime.hpp"

/* Typically used convention for layers:
 * rows (shape[0]): input features
 * cols (shape[1]): output features
 */

namespace yann
{
    Parameter::Parameter(const cum::dim_t rows, const cum::dim_t cols, const cum::layout format)
        : Parameter(cum::Shape{rows, cols}, format)
    {
    }

    Parameter::Parameter(const cum::Shape& shape, const cum::layout format)
    {
        values   = cum::Tensor(shape, cum::default_type, format);
        gradient = cum::Tensor(shape, cum::default_type, format);
    }

    Parameter Parameter::Uniform(const cum::dim_t rows, const cum::dim_t cols, const cum::layout format)
    {
        return Uniform(cum::Shape{rows, cols}, format);
    }

    Parameter Parameter::Uniform(const cum::Shape& shape, const cum::layout format)
    {
        constexpr cum::cumeric_t ampl = 0.5;

        Parameter param;
        param.values   = cum::Tensor::Random(shape, -ampl, ampl, cum::default_type, format);
        param.gradient = cum::Tensor::Zeros(shape, cum::default_type, format);

        return param;
    }

    Parameter Parameter::He(const cum::dim_t rows, const cum::dim_t cols, const cum::layout format)
    {
        return He(cum::Shape{rows, cols}, format);
    }

    Parameter Parameter::He(const cum::Shape& shape, const cum::layout format)
    {
        const cum::dim_t fan_in = shape.empty() ? 1 : shape.front();
        const cum::cumeric_t sigma = static_cast<cum::cumeric_t>(std::sqrt(2.0 / static_cast<double>(fan_in)));

        Parameter param;
        param.values = cum::Tensor(shape, cum::default_type, format);
        cum::random::normal(param.values.data<cum::cumeric_t>(), param.values.size(), sigma);
        param.gradient = cum::Tensor::Zeros(shape, cum::default_type, format);

        return param;
    }

    Parameter Parameter::Zeros(const cum::dim_t rows, const cum::dim_t cols, const cum::layout format)
    {
        return Zeros(cum::Shape{rows, cols}, format);
    }

    Parameter Parameter::Zeros(const cum::Shape& shape, const cum::layout format)
    {
        Parameter param;
        param.values   = cum::Tensor::Zeros(shape, cum::default_type, format);
        param.gradient = cum::Tensor::Zeros(shape, cum::default_type, format);

        return param;
    }

    Parameter Parameter::Ones(const cum::dim_t rows, const cum::dim_t cols, const cum::layout format)
    {
        return Ones(cum::Shape{rows, cols}, format);
    }

    Parameter Parameter::Ones(const cum::Shape& shape, const cum::layout format)
    {
        Parameter param;
        param.values   = cum::Tensor::Ones(shape, cum::default_type, format);
        param.gradient = cum::Tensor::Zeros(shape, cum::default_type, format);

        return param;
    }

    Parameter Parameter::Xavier(const cum::dim_t rows, const cum::dim_t cols, const cum::layout format)
    {
        return Xavier(cum::Shape{rows, cols}, format);
    }

    Parameter Parameter::Xavier(const cum::Shape& shape, const cum::layout format)
    {
        const cum::dim_t fan_in  = shape.empty() ? 1 : shape.front();
        const cum::dim_t fan_out = shape.empty() ? 1 : shape.back();
        const cum::cumeric_t limit = static_cast<cum::cumeric_t>(std::sqrt(6.0 / static_cast<double>(fan_in + fan_out)));

        Parameter param;
        param.values   = cum::Tensor::Random(shape, -limit, limit, cum::default_type, format);
        param.gradient = cum::Tensor::Zeros(shape, cum::default_type, format);

        return param;
    }

    void Parameter::clear_gradient()
    {
        gradient.fill(static_cast<cum::cumeric_t>(0));
        // gradient *= 0.1;
    }

    void Parameter::scale_gradient(cum::cumeric_t scalar)
    {
        gradient = gradient * scalar;
    }

    Parameter Parameter::FromSafetensors(const utils::SafeTensors& tensors, const std::string_view name, const bool load_gradient)
    {
        const std::string key(name);
        auto it = tensors.find(key);
        if (it == tensors.end())
        {
            throw std::runtime_error("Parameter::FromSafetensors: tensor '" + key + "' not found");
        }

        Parameter param;
        param.values = it->second;

        const std::string grad_key = key + ".grad";
        if (load_gradient)
        {
            if (auto grad_it = tensors.find(grad_key); grad_it != tensors.end())
            {
                if (grad_it->second.shape() != param.values.shape())
                {
                    throw std::runtime_error("Parameter::FromSafetensors: gradient shape mismatch for '" + grad_key + "'");
                }
                param.gradient = grad_it->second;
                return param;
            }
        }

        param.gradient = cum::Tensor::Zeros(param.values.shape(), param.values.type(), param.values.format());
        return param;
    }

    Parameter Parameter::FromSafetensors(const std::string& filename, const std::string_view name, const bool load_gradient)
    {
        const utils::SafeTensors tensors = utils::load_safetensors(filename);
        return FromSafetensors(tensors, name, load_gradient);
    }

    void Parameter::to_safetensors(utils::SafeTensors& tensors, const std::string_view name, const bool include_gradient) const
    {
        const std::string key(name);
        tensors.insert_or_assign(key, values);
        if (include_gradient)
        {
            tensors.insert_or_assign(key + ".grad", gradient);
        }
    }

    void Parameter::save_safetensors(const std::string& filename, const bool include_gradient, const std::string_view name) const
    {
        utils::SafeTensors tensors;
        to_safetensors(tensors, name, include_gradient);
        utils::save_safetensors(tensors, filename);
    }
} // yann

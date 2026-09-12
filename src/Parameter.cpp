//
// Created by jaro on 7/18/26.
//

#include "Parameter.hpp"

#include <cum/cum.hpp>
#include <cum/random.hpp>

#include <cmath>

#include "cum/memory.hpp"

/* Typically used convention for layers:
 * rows: output features
 * cols: input features
 */

namespace yann
{
    Parameter::Parameter(const cum::dim_t cols, const cum::dim_t rows) /* init with zeros by default */
    {
        // values   = cum::Tensor({rows, cols});
        // gradient = cum::Tensor({rows, cols});
        values   = cum::Tensor({rows, cols}, cum::default_type, cum::layout::IO);
        gradient = cum::Tensor({rows, cols});
    }

    Parameter::Parameter(const cum::Shape& shape)
    {
        values = cum::Tensor(shape, cum::default_type, cum::layout::IO);
        gradient = cum::Tensor(shape, cum::default_type, cum::layout::IO);
    }


    Parameter Parameter::Uniform(const cum::dim_t rows, const cum::dim_t cols)
    {
        constexpr cum::cumeric_t ampl = 0.1;

        Parameter param;
        param.values = cum::Tensor::Random({rows, cols}, -ampl, ampl, cum::default_type, cum::layout::IO);
        param.gradient = cum::Tensor::Zeros({rows, cols});
        return param;
    }

    Parameter Parameter::He(const cum::dim_t rows, const cum::dim_t cols)
    {
        cum::cumeric_t sigma = std::sqrt(2.f / (float)cols);

        Parameter param;
        param.values = cum::Tensor({rows, cols}, cum::default_type, cum::layout::IO);
        cum::random::normal(param.values.data(), rows * cols, sigma);
        param.gradient = cum::Tensor({rows, cols}, cum::default_type, cum::layout::IO);

        return param;
    }

    Parameter Parameter::Zeros(const cum::dim_t rows, const cum::dim_t cols)
    {
        Parameter param;
        param.values = cum::Tensor::Zeros({rows, cols}, cum::default_type, cum::layout::IO);
        param.gradient = cum::Tensor::Zeros({rows, cols}, cum::default_type, cum::layout::IO);
        return param;
    }

    Parameter Parameter::Ones(const cum::dim_t rows, const cum::dim_t cols)
    {
        Parameter param;
        param.values = cum::Tensor::Ones({rows, cols}, cum::default_type, cum::layout::IO);
        param.gradient = cum::Tensor::Zeros({rows, cols}, cum::default_type, cum::layout::IO);
        return param;
    }

    Parameter Parameter::Xavier(const cum::dim_t rows, const cum::dim_t cols)
    {

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
} // yann

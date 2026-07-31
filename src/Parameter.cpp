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
    Parameter::Parameter(std::size_t cols, std::size_t rows) /* init with zeros by default */
    {
        values   = cum::Matrix(rows, cols);
        gradient = cum::Matrix(rows, cols);
    }

    Parameter Parameter::Uniform(std::size_t rows, std::size_t cols)
    {
        constexpr cum::cumeric_t ampl = 0.1;

        Parameter param;
        param.values = cum::Matrix::Random(rows, cols, -ampl, ampl);
        param.gradient = cum::Matrix::Zeros(rows, cols);
        return param;
    }

    Parameter Parameter::He(std::size_t rows, std::size_t cols)
    {
        cum::cumeric_t sigma = std::sqrt(2.f / (float)cols);

        Parameter param;
        param.values = cum::Matrix(rows, cols);
        cum::random::normal(param.values.data(), rows * cols, sigma);
        param.gradient = cum::Matrix(rows, cols);

        return param;
    }

    Parameter Parameter::Zeros(std::size_t rows, std::size_t cols)
    {
        Parameter param;
        param.values = cum::Matrix::Zeros(rows, cols);
        param.gradient = cum::Matrix::Zeros(rows, cols);
        return param;
    }

    Parameter Parameter::Ones(std::size_t rows, std::size_t cols)
    {
        Parameter param;
        param.values = cum::Matrix::Ones(rows, cols);
        param.gradient = cum::Matrix(rows, cols);
        return param;
    }

    Parameter Parameter::Xavier(std::size_t rows, std::size_t cols)
    {

    }

    void Parameter::clear_gradient()
    {
        gradient.fill(static_cast<cum::cumeric_t>(0));
    }
} // yann
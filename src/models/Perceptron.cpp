#include "Perceptron.hpp"

#include <cum/LinearAlgebra.hpp>
#include <cum/functions.hpp>

namespace yann::models
{
    Perceptron::Perceptron(int inputLenght, const char* func) : _inputWidth_(inputLenght)
    {
        cum::functions::get_function_by_name(&activation, func);
        init();
    }

    cum::cumeric_t Perceptron::predict(const cum::Vector& input)
    {
        cum::cumeric_t result = (input * weights).sum() + bias;
        cum::functions::transformInPlace(&result, activation, 1);
        return result;
    }

    void Perceptron::fit(const cum::Matrix& input, const cum::Vector& target, cum::cumeric_t rate, size_t epochs)
    {
        for(size_t epoch = 0 ; epoch < epochs ; epoch++)
        {
            for(size_t row = 0 ; row < input.rows() ; row++)
            {
                cum::cumeric_t y = predict(input.row(row).flatten());
                cum::cumeric_t error = (target[row] - y) * rate;
                weights += (input.row(row).flatten() * error);
                bias += error;
            }
        }
    }

    void Perceptron::init()
    {

        weights = cum::Vector(_inputWidth_, 0.001);
        bias = 0;
    }
}
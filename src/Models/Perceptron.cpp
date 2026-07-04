#include "Perceptron.hpp"

#include "Utility/stlCompatibility.hpp"

#include <cum/LinearAlgebra.hpp>
#include <cum/functions.hpp>

namespace yann::models
{
    // Perceptron::Perceptron(int inputLenght, std::function<numeric_t(numeric_t)> func) : _inputWidth_(inputLenght)
    // {  
    //     activation = std::move(func);

    //     init();
    // }

    Perceptron::Perceptron(int inputLenght, const char* func) : _inputWidth_(inputLenght)
    {
        cum::functions::getFunctionByName(&activation, func);
        init();
    }

    cum::cumeric_t Perceptron::predict(const cum::Vector& input)
    {
        cum::cumeric_t result = (input * weights).sum() + bias;
        cum::functions::transformInPlace(&result, activation, 1);
        return result;
    }

    // numeric_t Perceptron::predict(const std::vector<numeric_t>& input)
    // {
    //     vector_t newInput = YANN::Utils::stlCompatibility::toEigenVector<numeric_t, std::vector<numeric_t>>(input);

    //     return predict(newInput);
    // }

    void Perceptron::fit(const cum::Matrix& input, const cum::Vector& target, cum::cumeric_t rate, size_t epochs)
    {
        for(size_t epoch = 0 ; epoch < epochs ; epoch++)
        {
            for(size_t row = 0 ; row < input.rows() ; row++)
            {
                cum::cumeric_t y = predict(input.row(row).flatten());

                // numeric_t error = (target[row] - y) * rate;
                // cum::cumeric_t error = static_cast<cum::cumeric_t>(cum::math_api::vectorAt(target, row) - y) * rate;
                cum::cumeric_t error = (target[row] - y) * rate;

                weights += (input.row(row).flatten() * error);
                bias += error;
            }
        }
    }

    // void Perceptron::fit(const std::vector<std::vector<numeric_t>>& input, const std::vector<numeric_t>& target, float rate, int epochs)
    // {
    //     // Eigen::MatrixXf newInput  = Utils::stlCompatibility::toEigenMatrix<numeric_t, Utils::stlCompatibility::STLMatrix<numeric_t>>(input);
    //     // Eigen::VectorXf newTarget = Utils::stlCompatibility::toEigenMatrix<numeric_t, std::vector<numeric_t>>(target);
    
    //     // train(newInput, newTarget, epochs, rate);
    // }

    void Perceptron::init()
    {
        // this->weights = vector_t::Random(_inputWidth_) * static_cast<numeric_t>(0.1f);
        // this->bias = static_cast<numeric_t>(0.0f);
        weights = cum::Vector(_inputWidth_, 0.001);
        bias = 0;
    }
}
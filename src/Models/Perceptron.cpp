#include "Perceptron.hpp"

#include "Utility/stlCompatibility.hpp"

namespace YANN::Models
{
    Perceptron::Perceptron(int inputLenght, std::function<numeric_t(numeric_t)> func) : _inputWidth_(inputLenght)
    {  
        activation = std::move(func);

        init();
    }

    numeric_t Perceptron::predict(const vector_t& input)
    {
        return activation((input * weights).sum() + bias);
        // return (input * weights).sum() + bias;
    }

    numeric_t Perceptron::predict(const std::vector<numeric_t>& input)
    {
        vector_t newInput = YANN::Utils::stlCompatibility::toEigenVector<numeric_t, std::vector<numeric_t>>(input);

        return predict(newInput);
    }

    void Perceptron::fit(const matrix_t& input, const vector_t& target, float rate, int epochs)
    {
        for(int epoch = 0 ; epoch < epochs ; epoch++)
        {
            for(int row = 0 ; row < input.rows() ; row++)
            {
                numeric_t y = predict(input.row(row).transpose());

                numeric_t error = (target[row] - y) * rate;

                weights += error * input.row(row).transpose();
                bias += error;
            }
        }
    }

    void Perceptron::fit(const std::vector<std::vector<numeric_t>>& input, const std::vector<numeric_t>& target, float rate, int epochs)
    {
        // Eigen::MatrixXf newInput  = Utils::stlCompatibility::toEigenMatrix<numeric_t, Utils::stlCompatibility::STLMatrix<numeric_t>>(input);
        // Eigen::VectorXf newTarget = Utils::stlCompatibility::toEigenMatrix<numeric_t, std::vector<numeric_t>>(target);
    
        // train(newInput, newTarget, epochs, rate);
    }

    void Perceptron::init()
    {
        this->weights = vector_t::Random(_inputWidth_) * static_cast<numeric_t>(0.1f);
        this->bias = 0;
    }
}
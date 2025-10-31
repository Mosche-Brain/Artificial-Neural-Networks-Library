#include "Perceptron.hpp"

#include "Utility/stlCompatibility.hpp"

namespace ANN::Models
{
    Perceptron::Perceptron(int inputLenght, std::function<float_t(float_t)> func) : _inputWidth_(inputLenght)
    {  
        activation = std::move(func);

        init();
    }

    float_t Perceptron::predict(const Eigen::VectorXf& input)
    {
        return activation((input * weights).sum() + bias);
        // return (input * weights).sum() + bias;
    }

    float_t Perceptron::predict(const std::vector<float_t>& input)
    {
        Eigen::VectorXf newInput = ANN::Utils::stlCompatibility::toEigenVector<float_t, std::vector<float_t>>(input);

        return predict(newInput);
    }

    void Perceptron::fit(const Eigen::MatrixXf& input, const Eigen::VectorXf& target, float rate, int epochs)
    {
        for(int epoch = 0 ; epoch < epochs ; epoch++)
        {
            for(int row = 0 ; row < input.rows() ; row++)
            {
                float_t y = predict(input.row(row).transpose());

                float error = (target[row] - y) * rate;

                weights += error * input.row(row);
                bias += error;
            }
        }
    }

    void Perceptron::fit(const std::vector<std::vector<float_t>>& input, const std::vector<float_t>& target, float rate, int epochs)
    {
        // Eigen::MatrixXf newInput  = Utils::stlCompatibility::toEigenMatrix<float_t, Utils::stlCompatibility::STLMatrix<float_t>>(input);
        // Eigen::VectorXf newTarget = Utils::stlCompatibility::toEigenMatrix<float_t, std::vector<float_t>>(target);
    
        // train(newInput, newTarget, epochs, rate);
    }

    void Perceptron::init()
    {
        this->weights = Eigen::VectorXf::Random(_inputWidth_) * 0.1f;
        this->bias = 0;
    }
}
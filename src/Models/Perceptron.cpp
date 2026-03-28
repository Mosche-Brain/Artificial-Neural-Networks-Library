#include "Perceptron.hpp"

#include "Utility/stlCompatibility.hpp"

namespace YANN::Models
{
    Perceptron::Perceptron(int inputLenght, std::function<f_type(f_type)> func) : _inputWidth_(inputLenght)
    {  
        activation = std::move(func);

        init();
    }

    f_type Perceptron::predict(const vector_t& input)
    {
        return activation((input * weights).sum() + bias);
        // return (input * weights).sum() + bias;
    }

    f_type Perceptron::predict(const std::vector<f_type>& input)
    {
        vector_t newInput = YANN::Utils::stlCompatibility::toEigenVector<f_type, std::vector<f_type>>(input);

        return predict(newInput);
    }

    void Perceptron::fit(const matrix_t& input, const vector_t& target, float rate, int epochs)
    {
        for(int epoch = 0 ; epoch < epochs ; epoch++)
        {
            for(int row = 0 ; row < input.rows() ; row++)
            {
                f_type y = predict(input.row(row).transpose());

                f_type error = (target[row] - y) * rate;

                weights += error * input.row(row).transpose();
                bias += error;
            }
        }
    }

    void Perceptron::fit(const std::vector<std::vector<f_type>>& input, const std::vector<f_type>& target, float rate, int epochs)
    {
        // Eigen::MatrixXf newInput  = Utils::stlCompatibility::toEigenMatrix<f_type, Utils::stlCompatibility::STLMatrix<f_type>>(input);
        // Eigen::VectorXf newTarget = Utils::stlCompatibility::toEigenMatrix<f_type, std::vector<f_type>>(target);
    
        // train(newInput, newTarget, epochs, rate);
    }

    void Perceptron::init()
    {
        this->weights = vector_t::Random(_inputWidth_) * static_cast<f_type>(0.1f);
        this->bias = 0;
    }
}
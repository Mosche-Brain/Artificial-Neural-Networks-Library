#pragma once

#include <Eigen/Dense>
#include <functional>

namespace SNN::Models
{
    class Perceptron
    {
    public:
        Perceptron(int inputLenght, const char* activation);
        Perceptron(int inputLenght, std::function<float(float)> activation);

        void train(Eigen::MatrixXf input, Eigen::VectorXf target, int epochs, float rate);
        
        float_t predict(Eigen::VectorXf input);

        int inputWidth() const { return _inputWidth_; }

    protected:
        void init();
        int _inputWidth_;

        Eigen::VectorXf weight;
        float_t bias;
    };
}
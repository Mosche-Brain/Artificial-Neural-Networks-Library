#pragma once

#include <Eigen/Dense>
#include <functional>

#include <vector>

#include "Utility/stlCompatibility.hpp"


namespace ANN::Models
{
    class Perceptron
    {
    public:
        Perceptron(int inputLenght, const char* func);
        Perceptron(int inputLenght, std::function<float_t(float_t)> func);

        void fit(const Eigen::MatrixXf& input, const Eigen::VectorXf& target, float rate, int epochs);
        void fit(const std::vector<std::vector<float_t>>& input, const std::vector<float_t>& target, float rate, int epochs);
        
        float_t predict(const Eigen::VectorXf& input);
        float_t predict(const std::vector<float_t>& input);

        int inputWidth() const { return _inputWidth_; }

    protected:
        void init();
        int _inputWidth_;

        float_t bias;
        Eigen::VectorXf weights;
        std::function<float_t(float_t)> activation;
    };
}
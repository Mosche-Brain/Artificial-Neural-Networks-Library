#pragma once

#include "build_config.hpp"
#include <Eigen/Dense>
#include <functional>

#include <vector>

#include "Utility/stlCompatibility.hpp"


namespace YANN::Models
{
    class Perceptron
    {
    public:
        Perceptron(int inputLenght, const char* func);
        Perceptron(int inputLenght, std::function<f_type(f_type)> func);

        void fit(const matrix_t& input, const vector_t& target, float rate, int epochs);
        void fit(const std::vector<std::vector<f_type>>& input, const std::vector<f_type>& target, float rate, int epochs);
        
        f_type predict(const vector_t& input);
        f_type predict(const std::vector<f_type>& input);

        int inputWidth() const { return _inputWidth_; }

    protected:
        void init();
        int _inputWidth_;

        f_type bias;
        vector_t weights;
        std::function<f_type(f_type)> activation;
    };
}
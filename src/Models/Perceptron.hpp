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
        Perceptron(int inputLenght, std::function<numeric_t(numeric_t)> func);

        void fit(const matrix_t& input, const vector_t& target, float rate, int epochs);
        void fit(const std::vector<std::vector<numeric_t>>& input, const std::vector<numeric_t>& target, numeric_t rate, int epochs);
        
        numeric_t predict(const vector_t& input);
        numeric_t predict(const std::vector<numeric_t>& input);

        int inputWidth() const { return _inputWidth_; }

    protected:
        void init();
        int _inputWidth_;

        numeric_t bias;
        vector_t weights;
        std::function<numeric_t(numeric_t)> activation;
    };
}
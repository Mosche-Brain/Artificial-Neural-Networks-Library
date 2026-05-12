#pragma once

#include <functional>

#include <vector>

#include "Utility/stlCompatibility.hpp"
#include "cum/cum.hpp"


namespace YANN::Models
{
    class Perceptron
    {
    public:
        Perceptron(int inputLenght, const char* func);
        Perceptron(int inputLenght, std::function<numeric_t(numeric_t)> func);

        void fit(const matrix_t& input, const vector_t& target, numeric_t rate, int epochs);
        
        cum::cumeric_t predict(const vector_t& input);

        int inputWidth() const { return _inputWidth_; }

    protected:
        void init();
        int _inputWidth_;

        cum::cumeric_t bias;
        vector_t weights;
        std::function<numeric_t(numeric_t)> activation;
    };
}
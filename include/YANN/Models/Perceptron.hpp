#pragma once

#include <functional>

#include <vector>

#include "cum/Matrix.hpp"
#include "cum/Vector.hpp"
#include "cum/functions.hpp"


namespace YANN::Models
{
    class Perceptron
    {
    public:
        Perceptron(int inputLenght, const char* func);
        // Perceptron(int inputLenght, std::function<numeric_t(numeric_t)> func);
        // Perceptron(int inputLenght, std::function<numeric_t(numeric_t)> func);

        void fit(const cum::Matrix& input, const cum::Vector& target, cum::cumeric_t rate, size_t epochs);
        
        cum::cumeric_t predict(const cum::Vector& input);

        int inputWidth() const { return _inputWidth_; }

    protected:
        void init();
        int _inputWidth_;

        cum::cumeric_t bias;
        cum::Vector weights;

        cum::functions::activation_t activation;
    };
}
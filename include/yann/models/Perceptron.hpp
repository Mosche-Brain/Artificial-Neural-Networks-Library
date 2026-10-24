/**
 * @file Perceptron.hpp
 * @brief Single-layer perceptron model implementation for linear classification and regression.
 */

#pragma once

#include <functional>
#include <vector>

#include "cum/Matrix.hpp"
#include "cum/Vector.hpp"
#include "cum/functions.hpp"

namespace yann::models
{
    /**
     * @class Perceptron
     * @brief Classical single-layer perceptron model.
     *
     * Computes output:
     * @f[ y = \sigma(w^T x + b) @f]
     */
    class Perceptron
    {
    public:
        /**
         * @brief Construct a Perceptron with given input dimension and activation function.
         * @param inputLenght Number of input features.
         * @param func Name of activation function (e.g. "sigmoid", "relu", "step").
         */
        Perceptron(int inputLenght, const char* func);
        // Perceptron(int inputLenght, std::function<numeric_t(numeric_t)> func);
        // Perceptron(int inputLenght, std::function<numeric_t(numeric_t)> func);

        /**
         * @brief Train the perceptron using gradient descent on the provided dataset.
         * @param input Matrix of input samples (rows: samples, cols: features).
         * @param target Vector of target labels.
         * @param rate Learning rate multiplier.
         * @param epochs Number of training epochs.
         */
        void fit(const cum::Matrix& input, const cum::Vector& target, cum::cumeric_t rate, size_t epochs);
        
        /**
         * @brief Predict output for a single input vector.
         * @param input Input vector of length matching inputWidth().
         * @return Predicted scalar output.
         */
        cum::cumeric_t predict(const cum::Vector& input);

        /**
         * @brief Get the expected number of input features.
         * @return Number of input features.
         */
        int inputWidth() const { return _inputWidth_; }

    protected:
        /**
         * @brief Initialize weight vector and bias parameters.
         */
        void init();

        int _inputWidth_;                        /**< Number of input features. */
        cum::cumeric_t bias;                     /**< Scalar bias term. */
        cum::Vector weights;                     /**< Weight vector for input features. */
        cum::functions::activation_t activation; /**< Selected activation function. */
    };
}

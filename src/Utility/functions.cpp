#include "functions.hpp"

#include <cmath>
#include <limits>

namespace YANN::Utils
{
    /*
    cum::cumeric_t linear(cum::cumeric_t x)
    {
        return x;
    }

    cum::cumeric_t linear_derivative(cum::cumeric_t x)
    {
        return 1;
    }

    cum::cumeric_t tanh(cum::cumeric_t x)
    {
        return tanhf(x);
    }

    cum::cumeric_t tanh_derivative(cum::cumeric_t x)
    {
        cum::cumeric_t tanh_x = tanhf(x);
        return 1.0f - tanh_x * tanh_x;          
    }

    cum::cumeric_t sigmoid(cum::cumeric_t x)
    {
        return 1.0f / (1.0f + expf(-x));
    }

    cum::cumeric_t sigmoid_derivative(cum::cumeric_t x)
    {
        cum::cumeric_t sigmoid_x = sigmoid(x);
        return sigmoid_x * (1 - sigmoid_x);
    }

    cum::cumeric_t ReLU(cum::cumeric_t x)
    {
        return std::max(static_cast<cum::cumeric_t>(0.0f), x);
    }

    cum::cumeric_t ReLU_derivative(cum::cumeric_t x)
    {
        return x < 0 ? 0 : 1;
    }
    
    #undef leaky_a
    #define leaky_a static_cast<cum::cumeric_t>(0.01f)
    cum::cumeric_t leaky_ReLU(cum::cumeric_t x)
    {
        return x < 0 ? x * leaky_a : x;
    }

    cum::cumeric_t leaky_ReLU_derivative(cum::cumeric_t x)
    {
        return x < 0 ? leaky_a : 1;
    }

    cum::cumeric_t GELU(cum::cumeric_t x) 
    {
        // Constants for GELU approximation
        const cum::cumeric_t sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
        const cum::cumeric_t coeff = 0.044715;

        // GELU approximation: x * sigmoid(sqrt(2/pi) * (x + 0.044715 * x^3))
        cum::cumeric_t x_cubed = x * x * x;
        cum::cumeric_t inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        cum::cumeric_t sigmoid = 1.0 / (1.0 + std::exp(-inner));
        return x * sigmoid;
    }

    cum::cumeric_t GELU_derivative(cum::cumeric_t x) 
    {
        // Constants for GELU approximation
        const cum::cumeric_t sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
        const cum::cumeric_t coeff = 0.044715;

        // Compute GELU components
        cum::cumeric_t x_cubed = x * x * x;
        cum::cumeric_t inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        cum::cumeric_t sigmoid = 1.0 / (1.0 + std::exp(-inner));

        // Derivative of sigmoid: sigmoid * (1 - sigmoid)
        cum::cumeric_t sigmoid_deriv = sigmoid * (1.0 - sigmoid);

        // Derivative of inner term: sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        cum::cumeric_t x_squared = x * x;
        cum::cumeric_t inner_deriv = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x_squared);

        // GELU derivative: sigmoid + x * sigmoid'(inner) * inner'
        return sigmoid + x * sigmoid_deriv * inner_deriv;
    }
    
    matrix_t softmax(const matrix_t& x)
    {
        matrix_t exp_input = (x.array() - x.maxCoeff()).exp();
        return exp_input / exp_input.sum();
    }

    matrix_t softmax_derivative(const matrix_t& x)
    {
        vector_t softmax_output = softmax(x);
    
        // int n = x.size();
        matrix_t softmax_diag = softmax_output.asDiagonal();
        matrix_t softmax_outer = softmax_output * softmax_output.transpose();
        
        return softmax_diag - softmax_outer;
    }
        */
}
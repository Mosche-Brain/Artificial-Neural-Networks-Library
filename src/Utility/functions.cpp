#include "functions.hpp"

namespace ANN::Utils
{
    float_t linear(float_t x)
    {
        return x;
    }

    float_t linear_derivative (float_t x)
    {
        return 1;
    }

    float_t tanh(float_t x)
    {
        return tanhf(x);
    }

    float_t tanh_derivative(float_t x)
    {
        return std::powf(acoshf(1.0 / x), 2.0f);
    }

    float_t sigmoid(float_t x)
    {
        return 1.0f / (1.0f + expf(-x));
    }

    float_t sigmoid_derivative(float_t x)
    {
        float_t sigma_x = sigmoid(x);
        return sigma_x * (1 - sigma_x);
    }

    float_t ReLU(float_t x)
    {
        return std::max(0.0f, x);
    }

    float_t ReLU_derivative(float_t x)
    {
        return x < 0 ? 0 : 1;
    }

    float_t GELU(float_t x) 
    {
        // Constants for GELU approximation
        const float_t sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
        const float_t coeff = 0.044715;

        // GELU approximation: x * sigmoid(sqrt(2/pi) * (x + 0.044715 * x^3))
        float_t x_cubed = x * x * x;
        float_t inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float_t sigmoid = 1.0 / (1.0 + std::exp(-inner));
        return x * sigmoid;
    }

    float_t GELU_derivative(float_t x) 
    {
        // Constants for GELU approximation
        const float_t sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
        const float_t coeff = 0.044715;

        // Compute GELU components
        float_t x_cubed = x * x * x;
        float_t inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float_t sigmoid = 1.0 / (1.0 + std::exp(-inner));

        // Derivative of sigmoid: sigmoid * (1 - sigmoid)
        float_t sigmoid_deriv = sigmoid * (1.0 - sigmoid);

        // Derivative of inner term: sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        float_t x_squared = x * x;
        float_t inner_deriv = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x_squared);

        // GELU derivative: sigmoid + x * sigmoid'(inner) * inner'
        return sigmoid + x * sigmoid_deriv * inner_deriv;
    }
    
    Eigen::MatrixXf softmax(const Eigen::MatrixXf& x)
    {
        Eigen::MatrixXf exp_input = (x.array() - x.maxCoeff()).exp();
        return exp_input / exp_input.sum();
    }

    Eigen::MatrixXf softmax_derivative(const Eigen::MatrixXf& x)
    {
        Eigen::VectorXf softmax_output = softmax(x);
    
        // int n = x.size();
        Eigen::MatrixXf softmax_diag = softmax_output.asDiagonal();
        Eigen::MatrixXf softmax_outer = softmax_output * softmax_output.transpose();
        
        return softmax_diag - softmax_outer;
    }
}
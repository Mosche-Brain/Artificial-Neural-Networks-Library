#include "loss.hpp"

#include <stdexcept>
#include <iostream>
#include <cmath>

namespace YANN::Utils::loss
{
    LossType computeLoss(const cum::Matrix& result, const cum::Matrix& target, LossFunction loss_function)
    {
        if(result.size() != target.size())
        {
            throw std::invalid_argument("result and target size doens't match");       
        }

        switch(loss_function)
        {
            case LossFunction::mse:
            {
                return mse(result, target);
            }
            case LossFunction::cross_entropy:
            {
                return cross_entropy(result, target);
            }
            case LossFunction::binary_cross_entropy:
            {
                return binary_cross_entropy(result, target);
            }
            default:
            {  
                throw std::invalid_argument("Unknown loss function");
            }
        }
    }


    LossType mse(const cum::Matrix& result, const cum::Matrix& target)
    {
        if (result.rows() != target.rows() || result.cols() != target.cols()) {
            throw std::runtime_error("Matrix dimensions do not match: result(" + 
                std::to_string(result.rows()) + "x" + std::to_string(result.cols()) + 
                "), target(" + std::to_string(target.rows()) + "x" + std::to_string(target.cols()) + ")");
        }

        if (result.cols() == 0) {
            throw std::runtime_error("Number of columns in result is zero");
        }

        // if (!result.allFinite() || !target.allFinite()) {
        //     throw std::runtime_error("Input matrices contain NaN or inf values");
        // }

        // Obliczenie różnicy
        cum::Matrix diff = result - target;

        cum::cumeric_t loss = diff.squaredNorm() / result.cols();
        cum::Matrix grad = diff / result.cols();

        return { loss, grad };


    }

    LossType binary_cross_entropy(const cum::Matrix& result, const cum::Matrix& target)
    {

        // cum::Matrix sigmoid_output = result.transform([](cum::cumeric_t x) { return 1.0f / (1.0f + std::exp(-x)); });

        // // Compute loss: -sum(targets * log(sigmoid_output) + (1 - targets) * log(1 - sigmoid_output)) / n_samples
        // cum::cumeric_t loss = static_cast<cumeric_t>(0.0f);
        // for (int j = 0; j < result.cols(); ++j) 
        // {
        //     for (int i = 0; i < result.rows(); ++i) 
        //     {
        //         loss -= target(i, j) * std::log(std::max(sigmoid_output(i, j), static_cast<cumeric_t>(1e-10f))) + 
        //                 (1 - target(i, j)) * std::log(std::max(1 - sigmoid_output(i, j), static_cast<cumeric_t>(1e-10f))); // Avoid log(0)
        //     }
        // }
        
        // loss /= result.cols();

        // // Gradient: sigmoid_output - targets
        // cum::Matrix d_result = sigmoid_output - target;
        // d_result /= static_cast<cumeric_t>(result.cols()); // Average over samples
        // return { loss, d_result };
    }

    LossType cross_entropy(const cum::Matrix& result, const cum::Matrix& target)
    {
        // cum::Matrix softmax_output = result.colwise().normalized().array().exp();
        // // softmax_output = softmax_output.array().rowwise() / softmax_output.array().colwise().sum();

        // // Compute loss: -sum(targets * log(softmax_output)) / n_samples
        // cum::cumeric_t loss = static_cast<cumeric_t>(0.0f);
        // for (int j = 0; j < result.cols(); ++j) 
        // {
        //     for (int i = 0; i < result.rows(); ++i) 
        //     {
        //         loss -= target(i, j) * std::log(std::max(softmax_output(i, j), static_cast<cumeric_t>(1e-10f))); // Avoid log(0)
        //     }
        // }
        
        // // loss = target.cwiseProduct(softmax_output.unaryExpr([](float& x){x = x * std::log(std::max(x, 1e-10f));})).sum() * -1;
        // loss /= result.cols();

        // // Gradient: softmax_output - targets
        // cum::Matrix d_result = softmax_output - target;
        // d_result /= static_cast<cum::cumeric_t>(result.cols()); // Average over samples
        // return { loss, d_result };
    }
}
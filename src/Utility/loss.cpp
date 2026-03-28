#include "loss.hpp"

#include <stdexcept>
#include <iostream>

namespace YANN::Utils::loss
{
    LossType computeLoss(const matrix_t& result, const matrix_t& target, LossFunction loss_function)
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


    LossType mse(const matrix_t& result, const matrix_t& target)
    {
        if (result.rows() != target.rows() || result.cols() != target.cols()) {
            throw std::runtime_error("Matrix dimensions do not match: result(" + 
                std::to_string(result.rows()) + "x" + std::to_string(result.cols()) + 
                "), target(" + std::to_string(target.rows()) + "x" + std::to_string(target.cols()) + ")");
        }

        if (result.cols() == 0) {
            throw std::runtime_error("Number of columns in result is zero");
        }

        if (!result.allFinite() || !target.allFinite()) {
            throw std::runtime_error("Input matrices contain NaN or inf values");
        }

        // Obliczenie różnicy
        matrix_t diff = result - target;

        f_type loss = diff.squaredNorm() / result.cols();
        matrix_t grad = diff / result.cols();

        return { loss, grad };


    }

    LossType binary_cross_entropy(const matrix_t& result, const matrix_t& target)
    {

        f_type epsilon = std::numeric_limits<f_type>::epsilon();
        Eigen::Array<f_type, Eigen::Dynamic, Eigen::Dynamic> p = result.array().max(epsilon).min(static_cast<f_type>(1.0f) - epsilon);  // clamping
        f_type loss = -(target.array() * p.log() + (static_cast<f_type>(1.0f) - target.array()) * (static_cast<f_type>(1.0f) - p).log()).mean();
        matrix_t grad = ((p - target.array()) / (p * (static_cast<f_type>(1.0f) - p)).max(epsilon)).matrix();
    
        return { loss, grad };
    }

    LossType cross_entropy(const matrix_t& result, const matrix_t& target)
    {
        matrix_t softmax_output = result.colwise().normalized().array().exp();
        softmax_output = softmax_output.array().rowwise() / softmax_output.array().colwise().sum();

        // Compute loss: -sum(targets * log(softmax_output)) / n_samples
        f_type loss = static_cast<f_type>(0.0f);
        for (int j = 0; j < result.cols(); ++j) 
        {
            for (int i = 0; i < result.rows(); ++i) 
            {
                loss -= target(i, j) * std::log(std::max(softmax_output(i, j), static_cast<f_type>(1e-10f))); // Avoid log(0)
            }
        }
        
        // loss = target.cwiseProduct(softmax_output.unaryExpr([](float& x){x = x * std::log(std::max(x, 1e-10f));})).sum() * -1;
        loss /= result.cols();

        // Gradient: softmax_output - targets
        matrix_t d_result = softmax_output - target;
        d_result /= static_cast<f_type>(result.cols()); // Average over samples
        return { loss, d_result };
    }
}
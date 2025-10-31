#include "loss.hpp"

#include <stdexcept>
#include <iostream>

namespace ANN::Utils::loss
{
    LossType computeLoss(const Eigen::MatrixXf& result, const Eigen::MatrixXf& target, LossFunction loss_function)
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
            default:
            {
                break;
            }
        }

        // return {0.0f, Eigen::MatrixXf::Zero()};
    }


    LossType mse(const Eigen::MatrixXf& result, const Eigen::MatrixXf& target)
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
        Eigen::MatrixXf diff = result - target;

        float loss = diff.squaredNorm() / result.cols();
        Eigen::MatrixXf grad = diff / result.cols();

        return { loss, grad };

        /*
        // Ograniczenie wartości w diff, aby uniknąć underflow/overflow
        const float max_value = 1e6f;
        const float min_value = 1e-20f;
        diff = diff.cwiseMax(-max_value).cwiseMin(max_value); // Ograniczenie dużych wartości
        diff = diff.cwiseMax(-min_value).cwiseMin(min_value); // Ograniczenie małych wartości

        // Sprawdzenie NaN w diff
        if (!diff.allFinite()) {
            throw std::runtime_error("NaN or inf in diff matrix");
        }

        // Obliczenie straty
        float squared_norm = diff.squaredNorm();
        if (squared_norm < 1e-30f) {
            squared_norm = 1e-30f; // Zapobieganie underflow
        }
        float loss = squared_norm / (2.0f * result.cols());
        if (std::isnan(loss) || std::isinf(loss)) {
            throw std::runtime_error("NaN or inf in loss");
        }

        // Obliczenie gradientu
        Eigen::MatrixXf d_result = diff / static_cast<float>(result.cols());
        if (!d_result.allFinite()) {
            throw std::runtime_error("NaN or inf in d_result");
        }

        return { loss, d_result };
        */

    }

    LossType cross_entropy(const Eigen::MatrixXf& result, const Eigen::MatrixXf& target)
    {
        Eigen::MatrixXf softmax_output = result.colwise().normalized().array().exp();
        softmax_output = softmax_output.array().rowwise() / softmax_output.array().colwise().sum();

        // Compute loss: -sum(targets * log(softmax_output)) / n_samples
        float_t loss = 0.0f;
        for (int j = 0; j < result.cols(); ++j) 
        {
            for (int i = 0; i < result.rows(); ++i) 
            {
                loss -= target(i, j) * std::log(std::max(softmax_output(i, j), 1e-10f)); // Avoid log(0)
            }
        }
        
        // loss = target.cwiseProduct(softmax_output.unaryExpr([](float& x){x = x * std::log(std::max(x, 1e-10f));})).sum() * -1;
        loss /= result.cols();

        // Gradient: softmax_output - targets
        Eigen::MatrixXf d_result = softmax_output - target;
        d_result /= static_cast<float>(result.cols()); // Average over samples
        return { loss, d_result };
    }
}
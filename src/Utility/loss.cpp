#include "loss.hpp"

#include <stdexcept>
#include <iostream>
#include <cmath>
#include <limits>

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
        if (result.rows() != target.rows() || result.cols() != target.cols())
        {
            throw std::invalid_argument(
                "binary_cross_entropy: result and target dimensions must match"
            );
        }

        const std::size_t rows = result.rows();
        const std::size_t cols = result.cols();
        const std::size_t size = result.size();

        // const cum::cumeric_t eps = cum::EPSILON;
        const cum::cummulative_t eps = std::numeric_limits<cum::cummulative_t>::epsilon();

        cum::cummulative_t loss = 0;
        cum::Matrix gradient(rows, cols, 0_c);

        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                const cum::cummulative_t y = (cum::cummulative_t)target(i, j);

                cum::cummulative_t p = (cum::cummulative_t)result(i, j);

                if (p < eps)
                    p = eps;
                else if (p > 1_c - eps)
                    p = 1 - eps;

                loss += -(
                    y * std::log(p)
                    + (1 - y) * std::log(1 - p)
                );

                gradient(i, j) =
                    (
                        (1_c - y) / (1_c - p)
                        - y / p
                    ) / static_cast<cum::cumeric_t>(size);
            }
        }

        loss /= static_cast<cum::cummulative_t>(size);

        return LossType{ loss, gradient };
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
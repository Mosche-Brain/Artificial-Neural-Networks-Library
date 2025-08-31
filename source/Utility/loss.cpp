#include "loss.hpp"

namespace SNN::Utils::loss
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

            // default:
            // {
            //     retur
            // }
        }
    }


    LossType mse(const Eigen::MatrixXf& result, const Eigen::MatrixXf& target)
    {
        Eigen::MatrixXf diff = result - target;
        float loss = diff.squaredNorm() / (2.0f * result.cols()); // Mean of squared errors
        Eigen::MatrixXf d_result = diff / static_cast<float>(result.cols()); // Gradient: (output - targets) / n_samples
        return { loss, d_result };
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
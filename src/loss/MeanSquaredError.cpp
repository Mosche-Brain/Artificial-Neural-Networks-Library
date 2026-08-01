//
// Created by jaro on 7/31/26.
//

#include "yann/loss/MeanSquaredError.hpp"

#include "cum/runtime.hpp"

namespace yann::loss
{
    void MeanSquaredError::compute(const cum::Matrix& predicted, const cum::Matrix& target)
    {
        constexpr bool fused_kernels = false;

        if constexpr (fused_kernels)
        {

        }
        else
        {
            // if (predicted.rows() != target.rows())
            if (predicted.rows() != target.rows() || predicted.cols() != target.cols())
            {
                throw std::runtime_error("Matrix dimensions do not match: result(" +
                    std::to_string(predicted.rows()) + "x" + std::to_string(predicted.cols()) +
                    "), target(" + std::to_string(target.rows()) + "x" + std::to_string(target.cols()) + ")");
            }

            if (predicted.cols() == 0)
                throw std::runtime_error("Number of columns in result is zero");

            const std::size_t N = predicted.rows();
            // if (!result.allFinite() || !target.allFinite()) {
            //     throw std::runtime_error("Input matrices contain NaN or inf values");
            // }

            cum::Matrix diff = predicted - target;

            loss.value = diff.squaredNorm() / static_cast<cum::cumeric_t>(N);
            loss.gradient = (diff * 2) / static_cast<cum::cumeric_t>(N);
        }
    }

    std::unique_ptr<LossBase> MeanSquaredError::create()
    {
        return std::make_unique<MeanSquaredError>();
    }
} // yann
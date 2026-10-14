//
// Created by jaro on 7/31/26.
//

#include <ranges>
#include <print>
#include <cum/runtime.hpp>

#include "yann/loss/MeanSquaredError.hpp"

#include "runtime_config.hpp"

namespace yann::loss
{
    void MeanSquaredError::compute_sum(const cum::Tensor& predicted, const cum::Tensor& target)
    {
        constexpr bool YANN_MSE_FUSED_PATH = false;

        if(runtime_config::fused_kernels())
        {
            if constexpr (YANN_MSE_FUSED_PATH)
            {
                // todo: implement
            }
            else
            {
                throw std::runtime_error("YANN_MSE_FUSED_PATH wasn't compiled");
            }
        }
        else
        {
            // sum over features and batch, scaling is applied by LossBase::compute
            cum::Tensor diff = predicted - target;

            loss.value = diff.squared_norm();

            loss.gradient = diff * 2;
        }
    }

    std::unique_ptr<LossBase> MeanSquaredError::create()
    {
        return std::make_unique<MeanSquaredError>();
    }
} // yann
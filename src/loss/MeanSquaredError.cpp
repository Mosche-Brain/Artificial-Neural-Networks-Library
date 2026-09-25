//
// Created by jaro on 7/31/26.
//

#include <ranges>
#include <print>
#include <cum/runtime.hpp>

#include "yann/loss/MeanSquaredError.hpp"
namespace yann::loss
{
    void MeanSquaredError::compute(const cum::Tensor& predicted, const cum::Tensor& target)
    {
        constexpr bool fused_kernels = false;

        if constexpr (fused_kernels)
        {

        }
        else
        {
            for (auto [index, axis] : predicted.shape() | std::ranges::views::enumerate)
                if (axis != target.shape()[index])
                    throw std::invalid_argument("MeanSquaredError::compute: result and target dimensions must match");


            const cum::dim_t N = predicted.lenght();

            cum::Tensor diff = predicted - target;

            loss.value = diff.squared_norm() / N;

            loss.gradient = (diff * 2) / N;
        }
    }

    std::unique_ptr<LossBase> MeanSquaredError::create()
    {
        return std::make_unique<MeanSquaredError>();
    }
} // yann
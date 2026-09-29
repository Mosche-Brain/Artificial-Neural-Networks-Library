/*
 * @author: jaro
 * @name:   CategoricalCrossEntropy
 * @file:   src/loss/CategoricalCrossEntropy.cpp
 * @date:   29 September 2026 18:50:55
 */

#include <ranges>

#include "yann/runtime_config.hpp"

#include "yann/loss/CategoricalCrossEntropy.hpp"

namespace yann::loss
{
    CategoricalCrossEntropy::CategoricalCrossEntropy(const bool from_logits) : from_logits_(from_logits)
    {

    }

    void CategoricalCrossEntropy::compute(const cum::Tensor& predicted, const cum::Tensor& target)
    {
        constexpr bool YANN_CCE_FUSED_PATH = false;

        if(runtime_config::fused_kernels())
        {
            if constexpr (YANN_CCE_FUSED_PATH)
            {
                // todo: implement
            }
            else
            {
                throw std::runtime_error("YANN_CCE_FUSED_PATH wasn't compiled");
            }
        }
        else
        {
            for (auto [index, axis] : predicted.shape() | std::ranges::views::enumerate)
                if (axis != target.shape()[index])
                    throw std::invalid_argument("CategoricalCrossEntropy::compute: result and target dimensions must match");

            constexpr cum::cumeric_t epsilon = 1e-5;

            loss.value = -target.multiply(predicted.clamp(epsilon, 1 - epsilon).log()).sum();

            loss.gradient = -(target / predicted);
        }
    }

    std::unique_ptr<CategoricalCrossEntropy> CategoricalCrossEntropy::create(const bool from_logits)
    {
        return std::make_unique<CategoricalCrossEntropy>(from_logits);
    }
} // yann::loss
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

    void CategoricalCrossEntropy::compute_sum(const cum::Tensor& predicted, const cum::Tensor& target)
    {
        constexpr bool YANN_CCE_FUSED_PATH = false;
        constexpr bool YANN_CCE_REFERENCE_PATH = true; // It will be moved to cmake config

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
            if constexpr (YANN_CCE_REFERENCE_PATH)
            {
                constexpr cum::cumeric_t epsilon = 1e-5;

                cum::Tensor P = predicted.clamp(epsilon, 1 - epsilon);

                loss.value = -target.multiply(P.log()).sum();

                loss.gradient = -(target / P);
            }
            else
            {
                throw std::runtime_error("YANN_CCE_REFERENCE_PATH wasn't compiled");
            }
        }
    }

    std::unique_ptr<CategoricalCrossEntropy> CategoricalCrossEntropy::create(const bool from_logits)
    {
        return std::make_unique<CategoricalCrossEntropy>(from_logits);
    }
} // yann::loss
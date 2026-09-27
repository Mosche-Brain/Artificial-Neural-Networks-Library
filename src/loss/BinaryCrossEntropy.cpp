//
// Created by jaro on 7/31/26.
//

#include <ranges>

#include "cum/functions/exponential.hpp"
#include "cum/functions/various.hpp"
#include "cum/neural_primitives/neural_kernels.hpp"

#include "yann/runtime_config.hpp"

#include "yann/loss/BinaryCrossEntropy.hpp"

namespace yann::loss
{
    void BinaryCrossEntropy::compute(const cum::Tensor& predicted, const cum::Tensor& target)
    {
        loss.value = 0;
        loss.gradient = cum::Tensor::Zeros(predicted.shape(), predicted.type(), predicted.format());

        if (runtime_config::fused_kernels())
        {
            throw std::runtime_error("Fused path for BCE weren't compiled");
        }
        else
        {
            for (auto [index, axis] : predicted.shape() | std::ranges::views::enumerate)
            {
                if (axis != target.shape()[index])
                    throw std::invalid_argument("binary_cross_entropy: result and target dimensions must match");
            }

            // const cum::cumeric_t batch_size = target.shape()[1];

            const cum::cumeric_t eps = 1e-6_c; // epsilon nie powinien być hardcoded wewnątrz funkcji, tylko raczej być globalnym makrem lub zmienną, ale nie dałem to na potrzeby debug

            const cum::Tensor P = predicted.clamp(eps, 1 - eps);

            // loss.value = (-1 * (target.multiply(P.log()) + (1 - target).multiply((1 - P).log()))).mean();
            loss.value = (-1 * (target.multiply(P.log()) + (1 - target).multiply((1 - P).log()))).sum();
            loss.gradient = ((1 - target) / (1 - P) - target / P);

        }
    }


    std::unique_ptr<LossBase> BinaryCrossEntropy::create()
    {
        return std::make_unique<BinaryCrossEntropy>();
    }
} // yann

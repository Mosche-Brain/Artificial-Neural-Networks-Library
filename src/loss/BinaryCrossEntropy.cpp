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

            const cum::cumeric_t batches = target.shape()[1];

            const cum::cumeric_t eps = 1e-5_c; // epsilon nie powinien być hardcoded wewnątrz funkcji, tylko raczej być globalnym makrem lub zmienną, ale nie dałem to na potrzeby debug

            const cum::Tensor P = predicted.clamp(eps, 1 - eps);

            loss.value = (-1 * (target.multiply(P.log()) + (1 - target).multiply((1 - P).log()))).mean();
            loss.gradient = ((1 - target) / (1 - P) - target / P) / batches;

            // cum::Shape indices(shape.size(), 0);
            // for (cum::dim_t axis = 0; axis < shape.size(); axis++)
            // {
            // for (cum::dim_t n = shape[axis] ; indices[axis] < n; indices[axis]++)
            // {
            // const cum::cumeric_t y = target.at(indices);
            // const cum::cumeric_t p = cum::functions::various::clamp(predicted.at(indices), eps, 1 - eps);
            //
            //
            // loss.value += -(y * cum::functions::exponential::log(p) + (1 - y) * cum::functions::exponential::log(1 - p));
            // loss.gradient.at(indices) = ((1 - y) / (1 - p) - y / p);
            // }
            // indices[axis] = 0;
            // }
        }
    }


    std::unique_ptr<LossBase> BinaryCrossEntropy::create()
    {
        return std::make_unique<BinaryCrossEntropy>();
    }
} // yann

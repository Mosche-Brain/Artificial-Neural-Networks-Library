//
// Created by jaro on 7/31/26.
//

#include <ranges>

#include "cum/functions/exponential.hpp"
#include "cum/functions/various.hpp"
#include "cum/neural_primitives/neural_kernels.hpp"

#include "yann/loss/BinaryCrossEntropy.hpp"

namespace yann::loss
{
    void BinaryCrossEntropy::compute(const cum::Tensor& predicted, const cum::Tensor& target)
    {
        constexpr bool fused_kernels = true;

        loss.value = 0;
        loss.gradient = cum::Matrix::Zeros(predicted.shape(), predicted.type(), predicted.format());

        if constexpr (fused_kernels)
        {
            // cum::neural_primitives::neural_kernels::BCE(loss.gradient.data(), &loss.value, predicted.data(), target.data(), predicted.rows(), predicted.cols());
        }
        else
        {
            // if (predicted.rows() != target.rows() || predicted.cols() != target.cols())
            for (auto [index, axis] : predicted.shape() | std::ranges::views::enumerate)
            {
                if (axis != target.shape()[index])
                throw std::invalid_argument(
                    "binary_cross_entropy: result and target dimensions must match"
                );
            }

            const cum::Shape& shape = predicted.shape();
            const cum::cumeric_t eps = 1e-7_c; // epsilon nie powinien być hardcoded wewnątrz funkcji, tylko raczej być globalnym makrem lub zmienną, ale nie dałem to na potrzeby debug

            cum::Shape indices(shape.size(), 0);
            for (cum::dim_t axis = 0; axis < shape.size(); axis++)
            {
                for (cum::dim_t n = shape[axis] ; indices[axis] < n; indices[axis]++)
                {
                    const cum::cumeric_t y = target.at(indices);
                    const cum::cumeric_t p = cum::functions::various::clamp(predicted.at(indices), eps, 1 - eps);


                    loss.value += -(y * cum::functions::exponential::log(p) + (1 - y) * cum::functions::exponential::log(1 - p));
                    loss.gradient.at(indices) = ((1 - y) / (1 - p) - y / p);
                }
                indices[axis] = 0;
            }

            // const std::size_t rows = predicted.rows();
            // const std::size_t cols = predicted.cols();
            // const std::size_t size = predicted.size();

        //     for (std::size_t i = 0; i < rows; ++i)  // iterowanie osobno po kolumnach i wierszach jest właściwie nie potrzebne, ale jeszcze nie chciiało mi się tego zmieniać
        //     {
        //         for (std::size_t j = 0; j < cols; ++j)
        //         {
        //             const cum::cumeric_t y = target(i, j);
        //             const cum::cumeric_t p = cum::functions::various::clamp(predicted(i, j), eps, 1 - eps);
        //
        //             loss.value += -(y * cum::functions::exponential::log(p) + (1 - y) * cum::functions::exponential::log(1 - p));
        //             loss.gradient(i, j) = ((1 - y) / (1 - p) - y / p);
        //         }
        //     }
        // }
        // loss.value /= predicted.cols();
    }


    std::unique_ptr<LossBase> BinaryCrossEntropy::create()
    {
        return std::make_unique<BinaryCrossEntropy>();
    }
} // yann

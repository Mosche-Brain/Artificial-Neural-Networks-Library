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
            // if (predicted.rows() != target.rows())
            // if (predicted.rows() != target.rows() || predicted.cols() != target.cols())
            // {
            //     throw std::runtime_error("Matrix dimensions do not match: result(" +
            //         std::to_string(predicted.rows()) + "x" + std::to_string(predicted.cols()) +
            //         "), target(" + std::to_string(target.rows()) + "x" + std::to_string(target.cols()) + ")");
            // }

            for (auto [index, axis] : predicted.shape() | std::ranges::views::enumerate)
                if (axis != target.shape()[index])
                    throw std::invalid_argument("MeanSquaredError::compute: result and target dimensions must match");


            // if (predicted.cols() == 0)
                // throw std::runtime_error("Number of columns in result is zero");

            // const std::size_t N = predicted.cols();
            const cum::dim_t N = predicted.lenght();
            // const std::size_t N = predicted.rows() * predicted.cols()                        ;
            // if (!result.allFinite() || !target.allFinite()) {
            //     throw std::runtime_error("Input matrices contain NaN or inf values");
            // }

            cum::Tensor diff = predicted - target;\
            std::println("sexo");
            cum::cumeric_t sample = diff.get_value(cum::Shape(diff.rank(), 0));
            std::println("sexy {}", sample);

            // loss.value = diff.squaredNorm();
            // loss.gradient = (diff * 2);
            loss.value = diff.lenght() > 1 ? diff.squaredNorm() : std::powf(diff.at(cum::Shape(diff.rank(), 0)), 2.f);// / static_cast<cum::cumeric_t>(N);
            std::println("sex");
            loss.gradient = (diff * 2) / static_cast<cum::cumeric_t>(N);
        }
    }

    std::unique_ptr<LossBase> MeanSquaredError::create()
    {
        return std::make_unique<MeanSquaredError>();
    }
} // yann
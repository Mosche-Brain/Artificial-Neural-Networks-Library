/**
 * @file Chain.hpp
 * @author jaro
 * @brief Non-owning view chaining multiple Sequential models into a single end-to-end pipeline.
 * @date 2026-10-24
 */

#ifndef YANN_CHAIN_HPP
#define YANN_CHAIN_HPP

#pragma once

#include <concepts>
#include <functional>
#include <initializer_list>
#include <span>
#include <vector>

#include "yann/Parameter.hpp"
#include "yann/logging/ITrainingCallback.hpp"
#include "yann/loss/LossBase.hpp"
#include "yann/models/Sequential.hpp"
#include "yann/optimizers/OptimizerBase.hpp"

namespace yann::models
{
    /**
     * @class Chain
     * @brief Non-owning view that chains two or more Sequential models together (e.g., Encoder -> Decoder).
     *
     * Forward propagation passes activations sequentially from the first model to the last.
     * Backward propagation flows gradients from the last model back to the first, accumulating
     * parameter gradients across all chained models so they can be trained jointly while
     * remaining independent Sequential instances.
     */
    class Chain
    {
    public:
        using ModelRef = std::reference_wrapper<Sequential>;

        Chain() = default;

        /**
         * @brief Construct a Chain from an initializer list of Sequential model references.
         * @param models Initializer list of Sequential references in execution order.
         */
        Chain(std::initializer_list<ModelRef> models);

        /**
         * @brief Variadic constructor accepting two or more Sequential models by reference.
         * @tparam Models Types convertible to Sequential&.
         * @param first First Sequential model in the chain.
         * @param second Second Sequential model in the chain.
         * @param rest Additional Sequential models in the chain.
         */
        template<typename... Models>
            requires (std::same_as<Models, Sequential> && ...)
        Chain(Sequential& first, Sequential& second, Models&... rest)
            : models_{std::ref(first), std::ref(second), std::ref(rest)...}
        {
        }

        /**
         * @brief Append a Sequential model reference to the end of the chain.
         * @param model Reference to a Sequential model.
         * @return Reference to this Chain for fluent chaining.
         */
        Chain& addModel(Sequential& model);

        /**
         * @brief Build all chained models by propagating output shapes from each model to the next.
         * @param input_shape Optional initial input shape for the first model.
         */
        void build(const cum::Shape& input_shape = {});

        /**
         * @brief Perform forward pass sequentially through all chained models.
         * @param input Input tensor fed into the first model.
         * @return Output tensor produced by the last model in the chain.
         */
        cum::Tensor forward(const cum::Tensor& input);

        /**
         * @brief Perform backward pass in reverse order through all chained models.
         * @param d_output Gradient with respect to the last model's output.
         * @return Gradient with respect to the first model's input.
         */
        cum::Tensor backward(const cum::Tensor& d_output);

        /**
         * @brief Jointly train all chained models end-to-end.
         * @param X Input tensor of shape (batch_size, ...).
         * @param Y Target tensor of shape (batch_size, ...).
         * @param loss Loss function evaluated on the last model's output.
         * @param optimizer Optimizer updating parameters across all chained models.
         * @param epochs Number of training epochs.
         * @param batch_size Mini-batch size (default: 1).
         * @param callbacks Optional training callbacks.
         */
        void fit(
            const cum::Tensor& X,
            const cum::Tensor& Y,
            loss::LossBase& loss,
            optimizers::OptimizerBase& optimizer,
            cum::dim_t epochs,
            cum::dim_t batch_size = 1,
            std::span<logging::ITrainingCallback*> callbacks = {}
        );

        /**
         * @brief Collect pointers to all trainable parameters across all chained models.
         * @return Vector of Parameter pointers from all models in the chain.
         */
        [[nodiscard]]
        std::vector<Parameter*> parameters() const;

        /**
         * @brief Get the number of models in the chain.
         * @return Count of chained Sequential models.
         */
        [[nodiscard]]
        std::size_t size() const noexcept { return models_.size(); }

        /**
         * @brief Check if the chain contains no models.
         * @return True if empty, false otherwise.
         */
        [[nodiscard]]
        bool empty() const noexcept { return models_.empty(); }

        /**
         * @brief Access a chained Sequential model by index.
         * @param index Position of the model in the chain.
         * @return Reference to the Sequential model at the given index.
         */
        Sequential& operator[](std::size_t index) const;

    private:
        std::vector<ModelRef> models_;
    };
} // namespace yann::models

#endif // YANN_CHAIN_HPP

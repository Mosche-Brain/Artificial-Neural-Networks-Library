/**
 * @file loss.hpp
 * @brief Umbrella header and legacy utility functions for loss computation in YANN.
 */

#pragma once 

#include <stdexcept>
#include <limits>

#include "yann/loss/LossBase.hpp"
#include "yann/loss/MeanAbsoluteError.hpp"
#include "yann/loss/MeanSquaredError.hpp"
#include "yann/loss/BinaryCrossEntropy.hpp"
#include "yann/loss/CategoricalCrossEntropy.hpp"

/* --------------------------------------- */

namespace yann::loss
{
    /**
     * @struct LossType
     * @brief Result structure for legacy matrix-based loss functions containing scalar loss and gradient matrix.
     */
    typedef struct
    {
        cum::cummulative_t loss;           /**< Calculated scalar loss value. */
        cum::cummulative_t gradient_scale; /**< Scaling factor applied to the gradient. */
        cum::Matrix gradient;              /**< Computed gradient with respect to output predictions. */
    } LossType;

    // using loss_t = std::pair<cum::cummulative_t, cum::Matrix>;
    
    /**
     * @enum LossFunction
     * @brief Supported legacy loss function identifiers.
     */
    enum class LossFunction 
    { 
        mse,                  /**< Mean Squared Error. */
        cross_entropy,        /**< Categorical Cross-Entropy. */
        binary_cross_entropy  /**< Binary Cross-Entropy. */
    };

    /**
     * @brief Compute the specified loss and its gradient for given predictions and targets.
     * @param result Model prediction matrix.
     * @param target Ground-truth target matrix.
     * @param loss_function Enum indicating which loss formulation to evaluate.
     * @return LossType structure with computed loss and gradient.
     */
    LossType computeLoss(cum::Matrix& result, cum::Matrix& target, LossFunction loss_function);

    /**
     * @brief Compute Mean Squared Error (MSE) loss and gradient on matrices.
     * @param result Model prediction matrix.
     * @param target Ground-truth target matrix.
     * @return LossType structure containing scalar MSE loss and gradient matrix.
     */
    LossType mse(const cum::Matrix& result, const cum::Matrix& target);

    // LossType binary_cross_entropy(const cum::Matrix& result, const cum::Matrix& target);

    /**
     * @brief Compute Binary Cross-Entropy (BCE) loss and gradient on matrices.
     * @param result Model prediction matrix.
     * @param target Ground-truth target matrix.
     * @return LossType structure containing scalar BCE loss and gradient matrix.
     */
    LossType binary_cross_entropy(cum::Matrix& result, cum::Matrix& target);

    /**
     * @brief Compute Categorical Cross-Entropy loss and gradient on matrices.
     * @param result Model prediction matrix.
     * @param target Ground-truth target matrix.
     * @return LossType structure containing scalar cross-entropy loss and gradient matrix.
     */
    LossType cross_entropy(const cum::Matrix& result, const cum::Matrix& target);
}

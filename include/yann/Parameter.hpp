/**
 * @file Parameter.hpp
 * @author jaro
 * @brief Trainable parameter encapsulation holding values and gradients for neural network layers.
 * @date 2026-07-18
 */

#ifndef YANN_PARAMERER_HPP
#define YANN_PARAMERER_HPP

#include <cum/Matrix.hpp>
#include "cum/Tensor.hpp"

namespace yann
{
    /**
     * @class Parameter
     * @brief Encapsulates a trainable model parameter (e.g. weights, biases) alongside its gradients.
     *
     * Holds the parameter's current tensor values and accumulated gradient tensor,
     * providing utility methods for shape queries, initialization strategies, and gradient updates.
     */
    class Parameter
    {
    public:
        cum::Tensor values;   /**< Tensor storing parameter values. */
        cum::Tensor gradient; /**< Tensor storing parameter gradients. */

        // cum::dim_t cols() const { return values.cols(); }
        // cum::dim_t rows() const { return values.rows(); }

        /**
         * @brief Get row count (first dimension of shape).
         * @return Number of rows in the parameter tensor.
         */
        [[nodiscard]]
        cum::dim_t rows() const { return values.shape()[0]; }

        /**
         * @brief Get column count (second dimension of shape).
         * @return Number of columns in the parameter tensor.
         */
        [[nodiscard]]
        cum::dim_t cols() const { return values.shape()[1]; }

        /**
         * @brief Get the shape of the parameter tensor.
         * @return Const reference or copy of tensor shape dimensions.
         */
        [[nodiscard]]
        cum::Shape shape() const { return values.shape(); }

        /**
         * @brief Get the underlying numerical data type.
         * @return cum::datatype enumeration value.
         */
        [[nodiscard]]
        cum::datatype type() const { return values.type(); }

        /**
         * @brief Get the memory layout of the parameter tensor.
         * @return cum::layout enumeration value.
         */
        [[nodiscard]]
        cum::layout format() const { return values.format(); }

        /**
         * @brief Reset the accumulated gradient tensor to zero.
         */
        void clear_gradient();

        /**
         * @brief Scale all gradient elements by a scalar value.
         * @param scalar Multiplicative scaling factor.
         */
        void scale_gradient(cum::cumeric_t scalar);

        /**
         * @brief Default constructor for an uninitialized parameter.
         */
        Parameter() = default;

        /**
         * @brief Construct a 2D parameter with given column and row dimensions.
         * @param cols Column dimension.
         * @param rows Row dimension.
         */
        Parameter(cum::dim_t cols, cum::dim_t rows);

        /**
         * @brief Construct a parameter with an arbitrary multi-dimensional shape.
         * @param shape Shape specification for the parameter.
         */
        Parameter(const cum::Shape& shape);

        /**
         * @brief Initialize a 2D parameter with uniform random values in [-0.1, 0.1].
         * @param rows Row dimension.
         * @param cols Column dimension.
         * @return Initialized Parameter instance.
         */
        static Parameter Uniform(cum::dim_t rows, cum::dim_t cols); // +-0.1

        /**
         * @brief Initialize a 2D parameter using He (Kaiming) initialization.
         * @param rows Row dimension.
         * @param cols Column dimension.
         * @return Initialized Parameter instance.
         */
        static Parameter He(cum::dim_t rows, cum::dim_t cols);

        /**
         * @brief Initialize a 2D parameter filled with zeros.
         * @param rows Row dimension.
         * @param cols Column dimension.
         * @return Initialized Parameter instance.
         */
        static Parameter Zeros(cum::dim_t rows, cum::dim_t cols);

        /**
         * @brief Initialize a 2D parameter filled with ones.
         * @param rows Row dimension.
         * @param cols Column dimension.
         * @return Initialized Parameter instance.
         */
        static Parameter Ones(cum::dim_t rows, cum::dim_t cols);

        /**
         * @brief Initialize a 2D parameter using Xavier (Glorot) initialization.
         * @param rows Row dimension.
         * @param cols Column dimension.
         * @return Initialized Parameter instance.
         */
        static Parameter Xavier(cum::dim_t rows, cum::dim_t cols);

        /**
         * @brief Initialize an N-D parameter with uniform random values in [-0.1, 0.1].
         * @param shape Shape of the tensor.
         * @return Initialized Parameter instance.
         */
        static Parameter Uniform(const cum::Shape& shape); // +-0.1

        /**
         * @brief Initialize an N-D parameter using He initialization.
         * @param shape Shape of the tensor.
         * @return Initialized Parameter instance.
         */
        static Parameter He(const cum::Shape& shape);

        /**
         * @brief Initialize an N-D parameter filled with zeros.
         * @param shape Shape of the tensor.
         * @return Initialized Parameter instance.
         */
        static Parameter Zeros(const cum::Shape& shape);

        /**
         * @brief Initialize an N-D parameter filled with ones.
         * @param shape Shape of the tensor.
         * @return Initialized Parameter instance.
         */
        static Parameter Ones(const cum::Shape& shape);

        /**
         * @brief Initialize an N-D parameter using Xavier initialization.
         * @param shape Shape of the tensor.
         * @return Initialized Parameter instance.
         */
        static Parameter Xavier(const cum::Shape& shape);

        // cum::Matrix& operator()() { return values; }

        /**
         * @brief Function call operator returning a reference to the parameter values tensor.
         * @return Reference to values tensor.
         */
        cum::Tensor& operator()() { return values; }
    };
} // yann

#endif //YANN_PARAMERER_HPP

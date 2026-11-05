/**
 * @file Parameter.hpp
 * @author jaro
 * @brief Trainable parameter encapsulation holding values and gradients for neural network layers.
 * @date 2026-07-18
 */

#ifndef YANN_PARAMERER_HPP
#define YANN_PARAMERER_HPP

#include <string>
#include <string_view>

#include <cum/Matrix.hpp>
#include <cum/Tensor.hpp>
#include <cum/layout.hpp>

#include "yann/utils/safetensors.hpp"

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
         * @brief Construct a 2D parameter with given row and column dimensions.
         * @param rows Row dimension.
         * @param cols Column dimension.
         * @param format Memory layout for the parameter tensor.
         */
        Parameter(cum::dim_t rows, cum::dim_t cols, cum::layout format = cum::layout::AB);

        /**
         * @brief Construct a parameter with an arbitrary multi-dimensional shape.
         * @param shape Shape specification for the parameter.
         * @param format Memory layout for the parameter tensor.
         */
        Parameter(const cum::Shape& shape, cum::layout format = cum::layout::ANY);

        /**
         * @brief Initialize a 2D parameter with uniform random values in [-0.1, 0.1].
         * @param rows Row dimension.
         * @param cols Column dimension.
         * @param format Memory layout for the parameter tensor.
         * @return Initialized Parameter instance.
         */
        static Parameter Uniform(cum::dim_t rows, cum::dim_t cols, cum::layout format = cum::layout::AB); // +-0.1

        /**
         * @brief Initialize a 2D parameter using He (Kaiming) initialization.
         * @param rows Row dimension.
         * @param cols Column dimension.
         * @param format Memory layout for the parameter tensor.
         * @return Initialized Parameter instance.
         */
        static Parameter He(cum::dim_t rows, cum::dim_t cols, cum::layout format = cum::layout::AB);

        /**
         * @brief Initialize a 2D parameter filled with zeros.
         * @param rows Row dimension.
         * @param cols Column dimension.
         * @param format Memory layout for the parameter tensor.
         * @return Initialized Parameter instance.
         */
        static Parameter Zeros(cum::dim_t rows, cum::dim_t cols, cum::layout format = cum::layout::AB);

        /**
         * @brief Initialize a 2D parameter filled with ones.
         * @param rows Row dimension.
         * @param cols Column dimension.
         * @param format Memory layout for the parameter tensor.
         * @return Initialized Parameter instance.
         */
        static Parameter Ones(cum::dim_t rows, cum::dim_t cols, cum::layout format = cum::layout::AB);

        /**
         * @brief Initialize a 2D parameter using Xavier (Glorot) initialization.
         * @param rows Row dimension.
         * @param cols Column dimension.
         * @param format Memory layout for the parameter tensor.
         * @return Initialized Parameter instance.
         */
        static Parameter Xavier(cum::dim_t rows, cum::dim_t cols, cum::layout format = cum::layout::AB);

        /**
         * @brief Initialize an N-D parameter with uniform random values in [-0.1, 0.1].
         * @param shape Shape of the tensor.
         * @param format Memory layout for the parameter tensor.
         * @return Initialized Parameter instance.
         */
        static Parameter Uniform(const cum::Shape& shape, cum::layout format = cum::layout::ANY); // +-0.1

        /**
         * @brief Initialize an N-D parameter using He initialization.
         * @param shape Shape of the tensor.
         * @param format Memory layout for the parameter tensor.
         * @return Initialized Parameter instance.
         */
        static Parameter He(const cum::Shape& shape, cum::layout format = cum::layout::ANY);

        /**
         * @brief Initialize an N-D parameter filled with zeros.
         * @param shape Shape of the tensor.
         * @param format Memory layout for the parameter tensor.
         * @return Initialized Parameter instance.
         */
        static Parameter Zeros(const cum::Shape& shape, cum::layout format = cum::layout::ANY);

        /**
         * @brief Initialize an N-D parameter filled with ones.
         * @param shape Shape of the tensor.
         * @param format Memory layout for the parameter tensor.
         * @return Initialized Parameter instance.
         */
        static Parameter Ones(const cum::Shape& shape, cum::layout format = cum::layout::ANY);

        /**
         * @brief Initialize an N-D parameter using Xavier initialization.
         * @param shape Shape of the tensor.
         * @param format Memory layout for the parameter tensor.
         * @return Initialized Parameter instance.
         */
        static Parameter Xavier(const cum::Shape& shape, cum::layout format = cum::layout::ANY);

        /**
         * @brief Load a parameter from a pre-loaded SafeTensors map.
         * @param tensors Map of named tensors loaded from a .safetensors file.
         * @param name Key name of the parameter values tensor in the map.
         * @param load_gradient If true, loads "<name>.grad" if present (or "gradient" when name is "values").
         *                      Otherwise initializes gradient with zeros matching the values shape.
         * @return Loaded Parameter instance.
         */
        static Parameter FromSafetensors(const utils::SafeTensors& tensors, std::string_view name = "values", bool load_gradient = false);

        /**
         * @brief Load a parameter directly from a .safetensors file.
         * @param filename Path to the .safetensors file.
         * @param name Key name of the parameter values tensor in the file.
         * @param load_gradient If true, also loads the gradient tensor if present in the file.
         * @return Loaded Parameter instance.
         */
        static Parameter FromSafetensors(const std::string& filename, std::string_view name = "values", bool load_gradient = false);

        /**
         * @brief Insert this parameter's values (and optionally gradient) into a SafeTensors map.
         * @param tensors Target map of named tensors.
         * @param name Key name under which the parameter values tensor is stored (gradient is stored under "<name>.grad").
         * @param include_gradient Whether to also serialize the gradient tensor.
         */
        void to_safetensors(utils::SafeTensors& tensors, std::string_view name = "values", bool include_gradient = false) const;

        /**
         * @brief Save this parameter's values (and optionally gradient) directly to a .safetensors file.
         * @param filename Destination path for the .safetensors file.
         * @param include_gradient Whether to also serialize the gradient tensor.
         * @param name Key name under which the parameter values tensor is stored (gradient is stored under "<name>.grad").
         */
        void save_safetensors(const std::string& filename, bool include_gradient = false, std::string_view name = "values") const;

        // cum::Matrix& operator()() { return values; }

        /**
         * @brief Function call operator returning a reference to the parameter values tensor.
         * @return Reference to values tensor.
         */
        cum::Tensor& operator()() { return values; }
    };
} // yann

#endif //YANN_PARAMERER_HPP

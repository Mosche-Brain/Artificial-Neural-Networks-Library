//
// Created by jaro on 7/18/26.
//

#ifndef YANN_PARAMERER_HPP
#define YANN_PARAMERER_HPP

#include <cum/Matrix.hpp>

#include "cum/Tensor.hpp"

namespace yann
{
    class Parameter
    {
    public:
        cum::Tensor values;
        cum::Tensor gradient;

        // cum::dim_t cols() const { return values.cols(); }
        // cum::dim_t rows() const { return values.rows(); }

        [[nodiscard]]
        cum::dim_t cols() const { return values.shape()[0]; }

        [[nodiscard]]
        cum::dim_t rows() const { return values.shape()[1]; }

        [[nodiscard]]
        cum::Shape shape() const { return values.shape(); }

        void clear_gradient();
        void scale_gradient(cum::cumeric_t scalar);

        Parameter() = default;
        Parameter(std::size_t cols, std::size_t rows);
        Parameter(const cum::Shape& shape);

        static Parameter Uniform(std::size_t rows, std::size_t cols); // +-0.1
        static Parameter He(std::size_t rows, std::size_t cols);
        static Parameter Zeros(std::size_t rows, std::size_t cols);
        static Parameter Ones(std::size_t rows, std::size_t cols);
        static Parameter Xavier(std::size_t rows, std::size_t cols);

        static Parameter Uniform(const cum::Shape& shape); // +-0.1
        static Parameter He(const cum::Shape& shape);
        static Parameter Zeros(const cum::Shape& shape);
        static Parameter Ones(const cum::Shape& shape);
        static Parameter Xavier(const cum::Shape& shape);

        // cum::Matrix& operator()() { return values; }
        cum::Tensor& operator()() { return values; }
    };
} // yann

#endif //YANN_PARAMERER_HPP

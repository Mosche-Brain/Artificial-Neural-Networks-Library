//
// Created by jaro on 7/18/26.
//

#ifndef YANN_PARAMERER_HPP
#define YANN_PARAMERER_HPP

#include <cum/Matrix.hpp>

namespace yann
{
    class Parameter
    {
    public:
        cum::Matrix values;
        cum::Matrix gradient;

        cum::dim_t cols() const { return values.cols(); }
        cum::dim_t rows() const { return values.rows(); }

        void clear_gradient();
        void scale_gradient(cum::cumeric_t scalar);

        Parameter() = default;
        Parameter(std::size_t cols, std::size_t rows);

        static Parameter Uniform(std::size_t rows, std::size_t cols); // +-0.1
        static Parameter He(std::size_t rows, std::size_t cols);
        static Parameter Zeros(std::size_t rows, std::size_t cols);
        static Parameter Ones(std::size_t rows, std::size_t cols);
        static Parameter Xavier(std::size_t rows, std::size_t cols);

        cum::Matrix& operator()() { return values; }
    };
} // yann

#endif //YANN_PARAMERER_HPP

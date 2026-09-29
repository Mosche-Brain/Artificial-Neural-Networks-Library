/*
 * @author: jaro
 * @name:   CategoricalCrossEntropy
 * @file:   include/yann/loss/CategoricalCrossEntropy.hpp
 * @date:   29 September 2026 18:50:55
 */

#ifndef YANN_CATEGORICALCROSSENTROPY_HPP
#define YANN_CATEGORICALCROSSENTROPY_HPP

#include "yann/loss/LossBase.hpp"

namespace yann::loss
{
    class CategoricalCrossEntropy
    {
    public:
        void compute(const cum::Tensor& predicted, const cum::Tensor& target) override;
        static std::unique_ptr<LossBase> create();
    };
} // yann

#endif //YANN_CATEGORICALCROSSENTROPY_HPP

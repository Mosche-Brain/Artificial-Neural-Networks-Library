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
    class CategoricalCrossEntropy : public LossBase
    {
    public:
        CategoricalCrossEntropy(bool from_logits = false);

        void compute(const cum::Tensor& predicted, const cum::Tensor& target) override;
        static std::unique_ptr<CategoricalCrossEntropy> create(bool from_logits = false);
    private:
        bool from_logits_;
    };
} // yann::loss

#endif //YANN_CATEGORICALCROSSENTROPY_HPP

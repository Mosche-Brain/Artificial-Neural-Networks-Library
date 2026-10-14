//
// Created by jaro on 7/31/26.
//

#ifndef YANN_MEANSQUAREDERROR_HPP
#define YANN_MEANSQUAREDERROR_HPP

#include "yann/loss/LossBase.hpp"
#include "cum/Matrix.hpp"

namespace yann::loss
{

    class MeanSquaredError : public LossBase
    {
    public:
        static std::unique_ptr<LossBase> create();

    protected:
        void compute_sum(const cum::Tensor& predicted, const cum::Tensor& target) override;
    };
} // yann

#endif //YANN_MEANSQUAREDERROR_HPP

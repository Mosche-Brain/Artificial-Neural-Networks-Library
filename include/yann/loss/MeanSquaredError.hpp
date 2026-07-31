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
        void compute(const cum::Matrix& predicted, const cum::Matrix& target) override;
        static std::unique_ptr<LossBase> create();
    };
} // yann

#endif //YANN_MEANSQUAREDERROR_HPP

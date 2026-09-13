//
// Created by jaro on 7/31/26.
//

#ifndef YANN_MEANABSOLUTEERROR_HPP
#define YANN_MEANABSOLUTEERROR_HPP
#include <memory>

#include "LossBase.hpp"
#include "cum/Matrix.hpp"

namespace yann::loss
{
    class MeanAbsoluteError : public LossBase
    {
    public:
        void compute(const cum::Tensor& predicted, const cum::Tensor& target) override;
        static std::unique_ptr<LossBase> create();
    };
} // yann

#endif //YANN_MEANABSOLUTEERROR_HPP

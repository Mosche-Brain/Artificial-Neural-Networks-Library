//
// Created by jaro on 7/31/26.
//

#ifndef YANN_BINARYCROSSENTROPY_HPP
#define YANN_BINARYCROSSENTROPY_HPP

#include "yann/loss/LossBase.hpp"

#include <memory>

namespace yann::loss
{

    class BinaryCrossEntropy : public LossBase
    {
    public:
        void compute(const cum::Matrix& predicted, const cum::Matrix& target) override;
        static std::unique_ptr<LossBase> create();
    };


} // yann

#endif //YANN_BINARYCROSSENTROPY_HPP

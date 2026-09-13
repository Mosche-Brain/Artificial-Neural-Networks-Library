//
// Created by jaro on 7/31/26.
//

#ifndef YANN_LOSSBASE_HPP
#define YANN_LOSSBASE_HPP

#include "cum/Core.hpp"
#include "cum/Matrix.hpp"
#include "cum/Tensor.hpp"

namespace yann::loss
{
    typedef struct
    {
        cum::cumeric_t value;
        cum::Tensor gradient;
    } loss_t;


    class LossBase
    {
    public:
        virtual ~LossBase() = default;
        virtual void compute(const cum::Tensor& predicted, const cum::Matrix& Tensor) = 0;
        // virtual void compute(const cum::Matrix& predicted, const cum::Matrix& target) = 0;


        const loss_t& result() const { return loss; }
        cum::cumeric_t getLoss() const { return loss.value; }
        cum::Tensor getGradient() const { return loss.gradient; }
    protected:
        loss_t loss;
    };

    using Loss = std::unique_ptr<LossBase>;

} // yann

#endif //YANN_LOSSBASE_HPP

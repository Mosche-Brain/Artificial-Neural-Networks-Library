/* Created by jaro on 7/17/26. */

#ifndef YANN_OPTIMIZERBASE_HPP
#define YANN_OPTIMIZERBASE_HPP

namespace cum { class Matrix; }

namespace yann::optimizers
{
    class OptimizerBase
    {
    public:
        virtual ~OptimizerBase() = default;
        virtual void step(cum::Matrix& params, cum::Matrix& grad) = 0;
    };
} // yann

#endif //YANN_OPTIMIZERBASE_HPP

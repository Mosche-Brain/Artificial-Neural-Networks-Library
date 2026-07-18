//
// Created by jaro on 7/17/26.
//

#include "optimizers/SGD.hpp"

namespace yann::optimizers
{
    SGD::SGD()
    {

    }

    SGD::SGD(cum::cumeric_t learning_rate) : learning_rate(learning_rate)
    {

    }

    void SGD::step(cum::Matrix& params, cum::Matrix& grad)
    {
        
    }

    std::unique_ptr<SGD> SGD::create(cum::cumeric_t learning_rate)
    {
        return std::make_unique<SGD>(learning_rate);
    }
} // yann
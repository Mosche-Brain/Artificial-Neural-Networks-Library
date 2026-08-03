//
// Created by jaro on 7/17/26.
//

#include "optimizers/SGD.hpp"

// #include <print>

#include "cum/runtime.hpp"

namespace yann::optimizers
{

    SGD::SGD(cum::cumeric_t learning_rate)  : OptimizerBase(learning_rate)
    {

    }

    void SGD::step(cum::Matrix& params, cum::Matrix& grad)
    {
        // tą funkcje prawdopodobnie zlikwiduje
    }

    void SGD::step(std::vector<Parameter*>& params)
    {
        // std::println("params count: {}", params.size());
        for(Parameter* param : params)
        {
            // std::println("updating param");
            param->values -= param->gradient * learning_rate;
            cum::runtime::sync();

            // add small random offset
            // cum::Matrix offset = cum::Matrix::Random(param->values.rows(), param->values.cols(), -0.01_c, 0.01_c);
            // cum::runtime::sync();

            // param->values += offset;
            // cum::runtime::sync();
            // param->values -= (param->gradient * learning_rate);
            // std::println("clearing grad");
            param->clear_gradient();
            // param->scale_gradient(0.999_c);
            cum::runtime::sync();
        }
        // learning_rate *= 0.99999f;
        // kod w komentarzach był do debugowania
    }

    std::unique_ptr<SGD> SGD::create(cum::cumeric_t learning_rate)
    {
        return std::make_unique<SGD>(learning_rate);
    }
} // yann
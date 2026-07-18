//
// Created by jaro on 7/17/26.
//

#ifndef YANN_SGD_HPP
#define YANN_SGD_HPP

#include "optimizers/OptimizerBase.hpp"

#include <cum/cum.hpp>
#include <vector>
#include <memory>

namespace yann::optimizers
{
    class SGD : public OptimizerBase
    {
    public:
        SGD();
        SGD(cum::cumeric_t learning_rate);

        // void step() override;
        void step(cum::Matrix& params, cum::Matrix& grad) override;

        static std::unique_ptr<SGD> create(cum::cumeric_t learning_rate);
    private:
        cum::cumeric_t learning_rate;
        // std::vector<cum::Matrix*> weights_grad;
        // std::vector<cum::Matrix*> biases_grad; // column vector
        std::vector<cum::Matrix> momentum;
        // std::vector<cum::Matrix*> weights;
        // std::vector<cum::Matrix*> biases; // column vector
    };
 } // yann

#endif //YANN_SGD_HPP

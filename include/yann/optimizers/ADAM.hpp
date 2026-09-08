//
// Created by jaro on 7/31/26.
//

#ifndef YANN_ADAM_HPP
#define YANN_ADAM_HPP

#include <unordered_map>

#include "yann/optimizers/OptimizerBase.hpp"

namespace yann::optimizers
{
    class ADAM : public OptimizerBase
    {
    public:
        ADAM(cum::cumeric_t learning_rate, cum::cumeric_t b1=0.9, cum::cumeric_t b2=0.999);

		void step(Parameter& param) override;
        void step(std::vector<Parameter*>& params) override;

		static std::unique_ptr<ADAM> create(cum::cumeric_t learning_rate, cum::cumeric_t b1=0.9, cum::cumeric_t b2=0.999);
    private:
		std::size_t current_step;
        cum::cumeric_t b1;
        cum::cumeric_t b2;
        // std::vector<cum::Matrix> momentum;
		// std::vector<cum::Matrix> second_momentum;
        std::unordered_map<Parameter*, cum::Matrix> momentum;
		std::unordered_map<Parameter*, cum::Matrix> second_momentum;
    };
} // yann

#endif //YANN_ADAM_HPP

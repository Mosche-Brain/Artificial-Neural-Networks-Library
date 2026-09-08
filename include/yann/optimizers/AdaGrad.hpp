#pragma once

#include <unordered_map>

#include "yann/optimizers/OptimizerBase.hpp"

namespace yann::optimizers
{
	class AdaGrad : public OptimizerBase
	{
	public:
		AdaGrad(cum::cumeric_t rate, cum::cumeric_t eps=1e-8);

		void step(Parameter& param) override;

		void step(std::vector<Parameter*>& params) override;

		std::unique_ptr<AdaGrad> create(cum::cumeric_t rate, cum::cumeric_t eps=1e-8);
	private:
		cum::cumeric_t eps;
		std::unordered_map<Parameter*, cum::Matrix> sum_sq_g;
	};
}

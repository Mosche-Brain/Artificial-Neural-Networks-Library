#include "yann/optimizers/AdaGrad.hpp"
#include "Parameter.hpp"

namespace yann::optimizers
{
	AdaGrad::AdaGrad(const cum::cumeric_t rate, const cum::cumeric_t eps) : OptimizerBase(rate), eps(eps)
	{

	}

	void AdaGrad::step(Parameter& param)
	{
		if(!sum_sq_g.contains(&param))
			sum_sq_g.emplace(&param, cum::Tensor(param.shape(), param.type(), param.format()).fill(eps));

		cum::Tensor& w = param.values;
		cum::Tensor& g = param.gradient;
		cum::Tensor& v = sum_sq_g.at(&param);

		v += g.square();

		w -= learning_rate / v.sqrt() * g;
		param.clear_gradient();
	}

	void AdaGrad::step(std::vector<Parameter*>& params)
	{
		OptimizerBase::step(params);
	}

	std::unique_ptr<AdaGrad> AdaGrad::create(const cum::cumeric_t rate, const cum::cumeric_t eps)
	{
		return std::make_unique<AdaGrad>(rate, eps);
	}
}

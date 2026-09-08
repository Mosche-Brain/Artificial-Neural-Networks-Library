#include "yann/optimizers/AdaGrad.hpp"
#include "Parameter.hpp"
#include "cum/Matrix.hpp"

namespace yann::optimizers
{
	AdaGrad::AdaGrad(const cum::cumeric_t rate, const cum::cumeric_t eps) : OptimizerBase(rate), eps(eps)
	{

	}

	void AdaGrad::step(Parameter& param)
	{
		if(!sum_sq_g.contains(&param))
			sum_sq_g.emplace(&param, cum::Matrix(param.rows(), param.cols(), eps));

		cum::Matrix& w = param.values;
		cum::Matrix& g = param.gradient;
		cum::Matrix& v = sum_sq_g[&param];

		v += g.square();

		w -= learning_rate / v.sqrt() * g;
	}

	void AdaGrad::step(std::vector<Parameter*>& params) // yes, this is redundant
	{
		if(sum_sq_g.empty())
			for(Parameter* param : params)
				if(!sum_sq_g.contains(param))
					sum_sq_g.emplace(param, cum::Matrix(param->rows(), param->cols(), eps));
		
		for(Parameter* param : params)
		{
			cum::Matrix& w = param->values;
			cum::Matrix& g = param->gradient;
			cum::Matrix& v = sum_sq_g[param];

			v += g.square();

			w -= learning_rate / v.sqrt() * g;
		}
	}

	std::unique_ptr<AdaGrad> create(const cum::cumeric_t rate, const cum::cumeric_t eps)
	{
		return std::make_unique<AdaGrad>(rate, eps);
	}
}

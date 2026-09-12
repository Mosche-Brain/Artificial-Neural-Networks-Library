/**----------------------------------------------
 * ?                    ABOUT
 * @author      : Świadomy Mężczyzna
 * @email       : 
 * @file        : src/optimizers/Momentum.cpp
 * @createdOn   : 09/08/2026
 * @description : Implementation of SGD with momentum optimizer class
 *---------------------------------------------**/


#include "yann/optimizers/Momentum.hpp"

namespace yann::optimizers
{
	Momentum::Momentum(const cum::cumeric_t rate, const cum::cumeric_t beta) : OptimizerBase(rate), b(beta)
	{

	}

	void Momentum::step(Parameter& param)
	{
		if(!momentum.contains(&param))
			momentum.emplace(&param, cum::Tensor::Zeros(param.shape(), param.type(), param.format()));
	}

	void Momentum::step(std::vector<Parameter*>& params)
	{
		if(momentum.empty())
			for(Parameter* param : params)
				if(!momentum.contains(param))
					momentum.emplace(param, cum::Tensor::Zeros(param->shape(), param->type(), param->format()));

		for(Parameter* param : params)
		{
			cum::Tensor& w = param->values;
			cum::Tensor& g = param->gradient;
			cum::Tensor& v = momentum.at(param);

			v = b * v + (1 - b) * g;

			w -= v;
		}
	}

	std::unique_ptr<Momentum> Momentum::create(const cum::cumeric_t rate, const cum::cumeric_t beta)
	{
		return std::make_unique<Momentum>(rate, beta);
	}
}

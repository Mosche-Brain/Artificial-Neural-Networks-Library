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
			momentum.emplace(&param, cum::Matrix::Zeros(param.rows(), param.cols()));
	}

	void Momentum::step(std::vector<Parameter*>& params)
	{
		if(momentum.empty())
			for(Parameter* param : params)
				if(!momentum.contains(param))
					momentum.emplace(param, cum::Matrix::Zeros(param->rows(), param->cols()));

		for(Parameter* param : params)
		{
			cum::Matrix& w = param->values;
			cum::Matrix& g = param->gradient;
			cum::Matrix& v = momentum[param];

			v = b * v + (1 - b) * g;

			w -= v;
		}
	}

	std::unique_ptr<Momentum> Momentum::create(const cum::cumeric_t rate, const cum::cumeric_t beta)
	{
		return std::make_unique<Momentum>(rate, beta);
	}
}

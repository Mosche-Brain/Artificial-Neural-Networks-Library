//
// Created by jaro on 7/31/26.
//

#include <ranges>
#include <cmath>

#include "yann/optimizers/ADAM.hpp"
#include "Parameter.hpp"
#include "cum/Core.hpp"
#include "cum/runtime.hpp"
#include "cum/cum.hpp"

namespace yann::optimizers
{
    ADAM::ADAM(cum::cumeric_t learning_rate, cum::cumeric_t b1, cum::cumeric_t b2) : OptimizerBase(learning_rate), b1(b1), b2(b2), current_step(0)
	{

	}

	void ADAM::step(Parameter& param)
	{
		if(!momentum.contains(&param))
			momentum.emplace(&param, cum::Matrix::Zeros(param.rows(), param.cols()));
		if(!second_momentum.contains(&param))
			second_momentum.emplace(&param, cum::Matrix::Zeros(param.rows(), param.cols()));

		current_step++;

		cum::Tensor& m = momentum.at(&param);
		cum::Tensor& v = second_momentum.at(&param);
		cum::Tensor& g = param.gradient;


		m = b1 * m + (1 - b1) * g;
		v = b2 * v + (1 - b2) * g.cwiseProduct(g);


		cum::Tensor mhat = m / (1 - std::pow(b1, current_step));
		cum::Tensor vhat = v / (1 - std::pow(b2, current_step));


		param.values -= learning_rate * mhat / (vhat.sqrt() + cum::EPSILON);
		param.clear_gradient();
	}

	void ADAM::step(std::vector<Parameter*>& params)
	{
		if(momentum.empty()) // first step
			for(auto* param : params)
			{
				if(!momentum.contains(param))
					momentum.emplace(param, cum::Tensor::Zeros(param->shape(), param->type(), param->format()));
				if(!second_momentum.contains(param))
					second_momentum.emplace(param, cum::Tensor::Zeros(param->shape(), param->type(), param->format()));
			}

		current_step++;
		for(auto [index, param] : params | std::views::enumerate)
		{
			cum::Tensor& m = momentum.at(param);
			cum::Tensor& v = second_momentum.at(param);
			cum::Tensor& g = param->gradient;


			m = b1 * m + (1 - b1) * g;
			v = b2 * v + (1 - b2) * g.cwiseProduct(g);


			cum::Tensor mhat = m / (1 - std::pow(b1, current_step));
			cum::Tensor vhat = v / (1 - std::pow(b2, current_step));


			param->values -= learning_rate * mhat / (vhat.sqrt() + cum::EPSILON);
			param->clear_gradient();
		}
	}

	std::unique_ptr<ADAM> ADAM::create(cum::cumeric_t learning_rate, cum::cumeric_t b1, cum::cumeric_t b2)
	{
        return std::make_unique<ADAM>(learning_rate, b1, b2);
	}
} // yann

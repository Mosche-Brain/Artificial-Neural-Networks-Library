
#include <cum/Matrix.hpp>
#include <cum/Vector.hpp>
#include <cum/cum.hpp>
#include <cum/Tensor.hpp>
#include <cum/neural_primitives/Primitive.hpp>

import yann;

int main()
{
	cum::cum(yann::DEVICE::GPU);

	yann::models::Sequential model({
		yann::models::layers::Input::createUnique({28, 28}),
		yann::models::layers::Conv2D::createUnique({3, 28, 28}, 3, {4, 4})
	});

	yann::runtime_config::set_verbosity(3);
	
	yann::Tensor temsor = yann::Tensor::Random({1, 3, 16, 16}, -1, 1, cum::datatype::FP16, cum::layout::NCHW);

	return 0;
}
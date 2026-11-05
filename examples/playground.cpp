
#include <cum/cum.hpp>

#include <yann/models/layers/Conv2D.hpp>

namespace cum::dnn
{
	using namespace cum::neural_primitives;
}

int main()
{
	cum::cum(cum::DEVICE::GPU);

	cum::dnn::Memory::Descriptor md({6, 7}, cum::datatype::U8, cum::layout::AB);

	yann::models::layers::Conv2D layer({3, 16, 16}, 3, {4, 4});

	cum::Tensor temsor = cum::Tensor::Random({1, 3, 16, 16}, -1, 1, cum::datatype::FP32, cum::layout::NCHW);

	layer.forward(temsor);

	return 0;
}
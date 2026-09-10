#include <cum/cum.hpp>
#include <cum/Matrix.hpp>
#include <cum/runtime.hpp>

#include <yann/models/Sequential.hpp>
#include <yann/optimizers/SGD.hpp>
#include <yann/optimizers/ADAM.hpp>
#include <yann/runtime_config.hpp>
#include <yann/loss/MeanSquaredError.hpp>
#include <yann/logging/LossTracker.hpp>
#include <yann/logging/GradientTracker.hpp>
#include <yann/utils/filesystem.hpp>

#include <matplot/matplot.h>
#include "helpers/conversion_helpers.hpp"

#include <iostream>
#include <numeric>


namespace plt = matplot;

int main()
{

    cum::cum(cum::DEVICE::CPU);


	cum::Matrix mat = cum::Matrix::Ones(6, 9);
	yann::utils::filesystem::dump_matrix(mat, "macierx.tensor");

	return 0;
	//
    yann::runtime_config::set_verbosity(0);

    yann::models::Sequential model({
         yann::models::layers::Input::createUnique(1),
         yann::models::layers::Dense::createUnique(64, "tanh"),
         yann::models::layers::Dense::createUnique(2, "linear"),
    });


    return 0;
}

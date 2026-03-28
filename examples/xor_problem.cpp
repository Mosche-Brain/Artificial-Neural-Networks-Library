#include <ANN/Models/Sequential.hpp>
#include "../src/build_config.hpp"

#include <iostream>

#include <matplot/matplot.h>

int main()
{
    YANN::Models::Sequential model
    ({
        YANN::Models::Layers::Input::createUnique(2),
        YANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(1, "tanh"),
    });

	matrix_t X(4, 2); 
	X << 0, 0,
         0, 1,
         1, 0,
         1, 1;

	vector_t Y(4);
	Y << 0, 1, 1, 0;

	std::cout << "train data:\n" << X << '\n'; 
	
	model.fit(X, Y, static_cast<f_type>(0.1f), 2000);

    for(int i = 0 ; i < 3 ; i++)
    {
        std::cout << i << " layer weights: "  << model.getWeights(i) << '\n';
    }

    std::cout << "\n=== Wyniki po treningu ===\n";
    for(int i = 0; i < 4; ++i)
    {
        matrix_t sample(2, 1);
        sample << X(i, 0), X(i, 1);

        matrix_t pred = model.forward(sample);
        f_type result = pred(0, 0);

        std::cout << "XOR(" << X(i, 0) << ", " << X(i, 1) << ") = "
                  << result << " → " << (result > 0.5 ? 1 : 0) << '\n';
    }

    matplot::fmesh([&](float x, float y) { 
        vector_t vec(2);
        vec(0) = static_cast<f_type>(x);
        vec(1) = static_cast<f_type>(y);
        return static_cast<float>(model.forward(vec)(0,0)); 
    });
    matplot::show();


    return 0;
}
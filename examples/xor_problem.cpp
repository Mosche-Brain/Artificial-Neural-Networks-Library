#include <YANN/Models/Sequential.hpp>
#include "../src/build_config.hpp"

#include <iostream>


int main()
{
    YANN::Models::Sequential model
    ({
        YANN::Models::Layers::Input::createUnique(2),
        YANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(1, "tanh"),
    });

	YANN::matrix_t X(4, 2); 
	X << 0, 0,
         0, 1,
         1, 0,
         1, 1;

	YANN::vector_t Y(4);
	Y << 0, 1, 1, 0;

	std::cout << "train data:\n" << X << '\n'; 
	
    int epochs = 2000;
    YANN::numeric_t learning_rate = (YANN::numeric_t)0.1f;
	model.fit(X, Y, learning_rate, epochs);

    for(size_t i = 0 ; i < 3 ; i++)
    {
        std::cout << i << " layer weights: "  << model.getWeights(i) << '\n';
    }

    std::cout << "\n=== Wyniki po treningu ===\n";
    for(size_t i = 0; i < 4; ++i)
    {
        YANN::matrix_t sample(2, 1);
        sample << X(i, 0), X(i, 1);

        YANN::matrix_t pred = model.forward(sample);
        YANN::numeric_t result = pred(0, 0);

        std::cout << "XOR(" << X(i, 0) << ", " << X(i, 1) << ") = "
                  << result << " → " << (result > 0.5 ? 1 : 0) << '\n';
    }

    // matplot::fmesh([&](float x, float y) { 
    //     YANN::vector_t vec(2);
    //     vec(0) = static_cast<YANN::numeric_t>(x);
    //     vec(1) = static_cast<YANN::numeric_t>(y);
    //     return static_cast<float>(model.forward(vec)(0,0)); 
    // });
    // matplot::show();


    return 0;
}
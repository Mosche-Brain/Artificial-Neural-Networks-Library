#include <ANN/Models/Sequential.hpp>

#include <iostream>

#include <matplot/matplot.h>

int main()
{
    ANN::Models::Sequential model
    ({
        ANN::Models::Layers::Input::createUnique(2),
        ANN::Models::Layers::Dense::createUnique(4, "relu"),
        ANN::Models::Layers::Dense::createUnique(1, "sigmoid"),
    });

	Eigen::MatrixXf X(4, 2); 
	X << 0, 0,
         0, 1,
         1, 0,
         1, 1;

	Eigen::VectorXf Y(4);
	Y << 0, 1, 1, 0;

	std::cout << "train data:\n" << X << '\n'; 
	
	model.fit(X, Y, 0.1f, 2000);

    for(int i = 0 ; i < 3 ; i++)
    {
        std::cout << i << " layer weights: "  << model.getWeights(i) << '\n';
    }

    std::cout << "\n=== Wyniki po treningu ===\n";
    for(int i = 0; i < 4; ++i)
    {
        Eigen::MatrixXf sample(2, 1);
        sample << X(i, 0), X(i, 1);

        Eigen::MatrixXf pred = model.forward(sample);
        float result = pred(0, 0);

        std::cout << "XOR(" << X(i, 0) << ", " << X(i, 1) << ") = "
                  << result << " → " << (result > 0.5 ? 1 : 0) << '\n';
    }

    // matplot::fmesh([&](float x, float y) { 
    //     Eigen::Vector2f vec{x, y};
    //     return model.forward(vec)(0,0); 
    // });
    // matplot::show();


    return 0;
}
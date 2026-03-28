#include <ANN/Models/Sequential.hpp>
#include "../src/build_config.hpp"

#include <iostream>

#include <matplot/matplot.h>

int main()
{
    YANN::Models::Sequential model
    ({
        YANN::Models::Layers::Input::createUnique(1),
        YANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(1, "tanh"),
    });

	vector_t X = vector_t::LinSpaced(200, static_cast<f_type>(-2 * M_PI), static_cast<f_type>(2 * M_PI));

	vector_t Y(200);
	for(int i = 0; i < 200; ++i)
    {
        Y(i) = std::sin(X(i));
    }

	// std::cout << "train data:\n" << X << '\n'; 
	
	model.fit(X, Y, static_cast<f_type>(0.1f), 50);

    // for(int i = 0 ; i < 3 ; i++)
    // {
    //     std::cout << i << " layer weights: "  << model.getWeights(i) << '\n';
    // }

    // plot model predictions vs true sine function
    vector_t Y_pred = model.forward(X);
    std::vector<double> x_data, y_data, y_pred_data;
    for(int i = 0; i < X.rows(); ++i)
    {
        x_data.push_back(X[i]);
        y_data.push_back(Y[i]);
        y_pred_data.push_back(Y_pred[i]);
    }
    matplot::figure();
    matplot::plot(x_data, y_data)->line_width(2).color("blue").display_name("True Sine");
    matplot::hold(matplot::on);
    matplot::plot(x_data, y_pred_data)->line_width(2).color("red").display_name("Model Prediction");
    matplot::title("Sine Function Approximation using Neural Network");
    matplot::xlabel("x");
    matplot::ylabel("sin(x)");
    matplot::legend();
    matplot::show();


    return 0;
}
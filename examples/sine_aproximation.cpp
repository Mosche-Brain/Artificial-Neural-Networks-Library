#include <ANN/Models/Sequential.hpp>

#include <iostream>

#include <matplot/matplot.h>

int main()
{
    ANN::Models::Sequential model
    ({
        ANN::Models::Layers::Input::createUnique(1),
        ANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        ANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        ANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        ANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        ANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        ANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        ANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        ANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        ANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        ANN::Models::Layers::Dense::createUnique(3, "leaky_relu"),
        ANN::Models::Layers::Dense::createUnique(1, "tanh"),
    });

	Eigen::VectorXf X(200);
    //setup linespace betwen -2pi to 2pi using built in eigen functions
    X = Eigen::VectorXf::LinSpaced(200, -2 * M_PI, 2 * M_PI);

	Eigen::VectorXf Y(200);
	for(int i = 0; i < 200; ++i)
    {
        Y(i) = std::sin(X(i));
    }

	// std::cout << "train data:\n" << X << '\n'; 
	
	model.fit(X, Y, 0.1f, 50);

    // for(int i = 0 ; i < 3 ; i++)
    // {
    //     std::cout << i << " layer weights: "  << model.getWeights(i) << '\n';
    // }

    // plot model predictions vs true sine function
    Eigen::VectorXf Y_pred = model.forward(X);
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
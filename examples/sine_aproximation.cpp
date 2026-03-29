#include <iostream>

#include <YANN/Models/Sequential.hpp>
#include <matplot/matplot.h>

void plot(YANN::vector_t x, YANN::vector_t y_true, YANN::vector_t y_pred);

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

    size_t num_samples = 200;
	YANN::vector_t X = YANN::vector_t::LinSpaced(num_samples, static_cast<YANN::numeric_t>(-2 * M_PI), static_cast<YANN::numeric_t>(2 * M_PI));

	YANN::vector_t Y(num_samples);
	for(size_t i = 0; i < num_samples; ++i)
    {
        Y(i) = std::sin(X(i));
    }

	size_t epochs = 50;
    YANN::numeric_t learning_rate = (YANN::numeric_t)0.1f;
	model.fit(X, Y, learning_rate, epochs);

  
    YANN::vector_t Y_pred = model.forward(X);
    
    plot(X, Y, Y_pred);

    return 0;
}

void plot(YANN::vector_t x, YANN::vector_t y_true, YANN::vector_t y_pred)
{
    std::vector<double> x_data, y_true_data, y_pred_data;
    for(size_t i = 0; i < x.rows(); ++i)
    {
        x_data.push_back(x[i]);
        y_true_data.push_back(y_true[i]);
        y_pred_data.push_back(y_pred[i]);
    }
    matplot::figure();
    matplot::plot(x_data, y_true_data)->line_width(2).color("blue").display_name("True Sine");
    matplot::hold(matplot::on);
    matplot::plot(x_data, y_pred_data)->line_width(2).color("red").display_name("Model Prediction");
    matplot::title("Sine Function Approximation using Neural Network");
    matplot::xlabel("x");
    matplot::ylabel("sin(x)");
    matplot::legend();
    matplot::show();
}
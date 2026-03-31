#include <iostream>
#include <fstream>

#include <YANN/Models/Sequential.hpp>
#include <matplot/matplot.h>

// void plot(YANN::vector_t x, YANN::vector_t y_true, YANN::vector_t y_pred);
void save(YANN::vector_t x, YANN::vector_t y_true, YANN::vector_t y_pretrain, YANN::vector_t y_pred);
// void plot(YANN::vector_t x, YANN::vector_t y_pred);

int main()
{
    YANN::Models::Sequential model
    ({
        YANN::Models::Layers::Input::createUnique(1),
        YANN::Models::Layers::Dense::createUnique(128, "relu"),
        YANN::Models::Layers::Dense::createUnique(1, "linear"),
    });

    size_t num_samples = 32;
	YANN::vector_t X = YANN::vector_t::LinSpaced(num_samples, static_cast<YANN::numeric_t>(-2 * M_PI), static_cast<YANN::numeric_t>(2 * M_PI));
    
	YANN::vector_t Y(num_samples);
	for(size_t i = 0; i < num_samples; ++i)
    {
        Y(i) = std::sin(X(i));
    }

    YANN::vector_t X_eval = YANN::vector_t::LinSpaced(256, static_cast<YANN::numeric_t>(-8 * M_PI), static_cast<YANN::numeric_t>(8 * M_PI));

	
    size_t epochs = 50;
    YANN::numeric_t learning_rate = (YANN::numeric_t)0.001f;
	
    YANN::vector_t Y_pretrain(YANN::math_api::vectorSize(X_eval));
    for(size_t i = 0 ; i < YANN::math_api::vectorSize(X_eval) ; i++)
    {
        Y_pretrain[i] = model.forward(YANN::math_api::createMatrix({{X_eval[i]}}))(0, 0);
    }
    
    model.fit(X, Y, learning_rate, epochs);

  
    YANN::vector_t Y_eval(YANN::math_api::vectorSize(X_eval));
    YANN::vector_t Y_true(YANN::math_api::vectorSize(X_eval));
    for(size_t i = 0 ; i < YANN::math_api::vectorSize(X_eval) ; i++)
    {
        Y_eval[i] = model.forward(YANN::math_api::createMatrix({{X_eval[i]}}))(0, 0);
        Y_true[i] = std::sinf(X_eval[i]);
    }
    // YANN::matrix_t Y_pred = model.forward(X);
    
    // plot(X_eval, Y, Y_eval);
    save(X_eval, Y_true, Y_pretrain, Y_eval);
    return 0;
}

void save(YANN::vector_t x, YANN::vector_t y_true, YANN::vector_t y_pretrain, YANN::vector_t y_pred)
{
    // save data to sifle, each vector as column, elements separated by comma, rows separated by newline
    std::ofstream file("sine_approximation_data.txt");
    if(file.is_open())
    {
        for(size_t i = 0; i < YANN::math_api::vectorSize(x); ++i)
        {
            file << std::to_string(x(i)) << " " << std::to_string(y_true(i)) << " " << std::to_string(y_pretrain(i)) << " " << std::to_string(y_pred(i)) << "\n";
        }
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing\n";
    }
}


// void plot(YANN::vector_t x, YANN::vector_t y_true, YANN::vector_t y_pred)
// {
//     std::vector<double> x_data, y_true_data, y_pred_data;
//     for(size_t i = 0; i < x.rows(); ++i)
//     {
//         x_data.push_back((double)(x[i]));
//         y_pred_data.push_back((double)(y_pred[i]));
//     }
//     matplot::figure();
//     matplot::plot(x_data, y_true_data)->line_width(2).color("blue").display_name("True Sine");
//     matplot::hold(matplot::on);
//     matplot::plot(x_data, y_pred_data)->line_width(2).color("red").display_name("Model Prediction");
//     matplot::title("Sine Function Approximation using Neural Network");
//     matplot::xlabel("x");
//     matplot::ylabel("sin(x)");
//     matplot::legend();
//     matplot::show();
// }
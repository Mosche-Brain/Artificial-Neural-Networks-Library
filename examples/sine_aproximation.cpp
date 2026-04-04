#include <iostream>
#include <fstream>

#include <YANN/Models/Sequential.hpp>
#include <matplot/matplot.h>

// void plot(YANN::vector_t x, YANN::vector_t y_true, YANN::vector_t y_pred);
// void save(YANN::vector_t row1, YANN::vector_t y_true, YANN::vector_t y_pretrain, YANN::vector_t y_pred);
void save(YANN::matrix_t mat, const char* filename);
// void plot(YANN::vector_t x, YANN::vector_t y_pred);

int main()
{
    YANN::math_api::setUsedThreadCount(8);

    YANN::Models::Sequential model_a
    ({
        YANN::Models::Layers::Input::createUnique(1),

        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(4, "tanh"),
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),

        YANN::Models::Layers::Dense::createUnique(1, "tanh"),
    });

    YANN::Models::Sequential model_b
    ({
        YANN::Models::Layers::Input::createUnique(1),
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(1, "linear"),
    });

    size_t num_samples = 84;
	YANN::vector_t X_train = YANN::vector_t::LinSpaced(num_samples, static_cast<YANN::numeric_t>(-6 * M_PI), static_cast<YANN::numeric_t>(6 * M_PI));
    
	YANN::vector_t Y_train(num_samples);
	for(size_t i = 0; i < num_samples; ++i)
    {
        Y_train(i) = std::sin(X_train(i));
    }

    YANN::vector_t X_eval = YANN::vector_t::LinSpaced(1024, static_cast<YANN::numeric_t>(-8 * M_PI), static_cast<YANN::numeric_t>(8 * M_PI));

	
    size_t epochs = 2000;
    YANN::numeric_t learning_rate = (YANN::numeric_t)0.0001f;
	
    YANN::vector_t Y_model_a_pretrain(YANN::math_api::vectorSize(X_eval));
    YANN::vector_t Y_model_b_pretrain(YANN::math_api::vectorSize(X_eval));
    for(size_t i = 0 ; i < YANN::math_api::vectorSize(X_eval) ; i++)
    {
        Y_model_a_pretrain[i] = model_a.forward(YANN::math_api::createMatrix({{X_eval[i]}}))(0, 0);
        Y_model_b_pretrain[i] = model_b.forward(YANN::math_api::createMatrix({{X_eval[i]}}))(0, 0);
    }
    
    model_a.fit(X_train, Y_train, learning_rate, epochs);
    model_b.fit(X_train, Y_train, learning_rate, epochs);

  
    YANN::vector_t Y_model_a_eval(YANN::math_api::vectorSize(X_eval));
    YANN::vector_t Y_model_b_eval(YANN::math_api::vectorSize(X_eval));
    YANN::vector_t Y_true(YANN::math_api::vectorSize(X_eval));
    for(size_t i = 0 ; i < YANN::math_api::vectorSize(X_eval) ; i++)
    {
        Y_model_a_eval[i] = model_a.forward(YANN::math_api::createMatrix({{X_eval[i]}}))(0, 0);
        Y_model_b_eval[i] = model_b.forward(YANN::math_api::createMatrix({{X_eval[i]}}))(0, 0);
        Y_true[i] = std::sinf(X_eval[i]);
    }
    // YANN::matrix_t Y_pred = model.forward(X);
    
    YANN::matrix_t traing_matrix(YANN::math_api::vectorSize(X_train), 2);
    for(size_t i = 0; i < YANN::math_api::vectorSize(X_train); ++i)
    {
        traing_matrix(i, 0) = X_train[i];
        traing_matrix(i, 1) = Y_train[i];
    }

    YANN::matrix_t results_matrix(YANN::math_api::vectorSize(X_eval), 6);
    for(size_t i = 0; i < YANN::math_api::vectorSize(X_eval); ++i)
    {
        results_matrix(i, 0) = X_eval[i];
        results_matrix(i, 1) = Y_true[i];
        results_matrix(i, 2) = Y_model_a_pretrain[i];
        results_matrix(i, 3) = Y_model_b_pretrain[i];
        results_matrix(i, 4) = Y_model_a_eval[i];
        results_matrix(i, 5) = Y_model_b_eval[i];
    }

    // plot(X_eval, Y_train, Y_eval);
    save(results_matrix, "sine_approximation_results.txt");
    save(traing_matrix, "sine_approximation_training_data.txt");
    return 0;
}

void save(YANN::matrix_t mat, const char* filename)
{
    std::ofstream file(filename);
    if(file.is_open())
    {
        for(size_t i = 0; i < YANN::math_api::matrixRows(mat); ++i)
        {
            for(size_t j = 0; j < YANN::math_api::matrixCols(mat); ++j)
            {
                file << std::to_string(mat(i, j)) << " ";
            }
            file << "\n";
            // file << std::to_string(x(i)) << " " << std::to_string(y_true(i)) << " " << std::to_string(y_pretrain(i)) << " " << std::to_string(y_pred(i)) << "\n";
        }
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing\n";
    }
}

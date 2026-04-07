#include <iostream>
#include <fstream>

#include <YANN/Models/Sequential.hpp>
#include <YANN/Utility/FileIO.hpp>
// #include <matplot/matplot.h>

// void plot(YANN::vector_t x, YANN::vector_t y_true, YANN::vector_t y_pred);
// void save(YANN::matrix_t mat, const char* filename);
// void plot(YANN::vector_t x, YANN::vector_t y_pred);

int main()
{
    YANN::math_api::setUsedThreadCount(8);

    YANN::Models::Sequential model
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
        YANN::Models::Layers::Dense::createUnique(16, "tanh"),
        YANN::Models::Layers::Dense::createUnique(1, "tanh"),
    });

    size_t num_samples = 84;
    int periods_for_training = 4;
    YANN::vector_t X_train = YANN::vector_t::LinSpaced(num_samples, (YANN::numeric_t)(-periods_for_training * 2 * M_PI), (YANN::numeric_t)(periods_for_training * 2 * M_PI));
    
	YANN::vector_t Y_train(num_samples);
	for(size_t i = 0; i < num_samples; ++i)
    {
        Y_train(i) = std::sin(X_train(i));
    }

    int periods_for_evaluation = 10;
    YANN::vector_t X_eval = YANN::vector_t::LinSpaced(1024, (YANN::numeric_t)(-periods_for_evaluation * 2 * M_PI), (YANN::numeric_t)(periods_for_evaluation * 2 * M_PI));
	
    size_t epochs = 2000;
    YANN::numeric_t learning_rate = (YANN::numeric_t)0.0001f;
	
    YANN::vector_t Y_pretrain(YANN::math_api::vectorSize(X_eval));
    for(size_t i = 0 ; i < YANN::math_api::vectorSize(X_eval) ; i++)
    {
        Y_pretrain[i] = model.forward(YANN::math_api::createMatrix({{X_eval[i]}}))(0, 0);
    }
    
    model.fit(X_train, Y_train, learning_rate, epochs);
  
    YANN::vector_t Y_eval(YANN::math_api::vectorSize(X_eval));
    YANN::vector_t Y_true(YANN::math_api::vectorSize(X_eval));
    for(size_t i = 0 ; i < YANN::math_api::vectorSize(X_eval) ; i++)
    {
        Y_eval[i] = model.forward(YANN::math_api::createMatrix({{X_eval[i]}}))(0, 0);
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
        results_matrix(i, 2) = Y_pretrain[i];
        results_matrix(i, 3) = Y_eval[i];
    }

    YANN::Utils::FileIO::saveMatrix(results_matrix, "sine_approximation_results.txt");
    YANN::Utils::FileIO::saveMatrix(traing_matrix, "sine_approximation_training_data.txt");

    YANN::Utils::FileIO::saveSequentialModel(model, "model_weights");

    return 0;
}

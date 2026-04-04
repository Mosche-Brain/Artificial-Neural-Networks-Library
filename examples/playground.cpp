#define ENABLE_DEBUG_OUTPUT
#include <iostream>
#include <YANN/Models/Sequential.hpp>
#include <YANN/Utility/FileIO.hpp>


int main()
{
    YANN::math_api::setUsedThreadCount(8);
    size_t hidden_layer_size = 10;
    YANN::Models::Sequential model({
        YANN::Models::Layers::Input::createUnique(3),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "tanh"),
        YANN::Models::Layers::Dense::createUnique(1, "linear")
    });

    YANN::matrix_t training_data;
    YANN::Utils::FileIO::readMatrix(training_data, "arithmatic-data.txt");

    // normalize training data
    YANN::math_api::matrixNormalizeInPlace(training_data);

    std::cout << "Training data shape: " << training_data.rows() << "x" << training_data.cols() << std::endl;

    // YANN::matrix_t X_train = training_data.block(0, 0, training_data.rows(), 2);
    YANN::matrix_t X_train = YANN::math_api::createMatrix(training_data.rows(), 3);
    // YANN::matrix_t Y_train = training_data.block(0, 3, training_data.rows(), 3);
    YANN::matrix_t Y_train = YANN::math_api::createMatrix(training_data.rows(), 1);

    for(size_t i = 0; i < training_data.rows(); ++i)
    {
        for(size_t j = 0; j < 3; ++j)
        {
            X_train(i, j) = training_data(i, j);
        }
        Y_train(i, 0) = training_data(i, 3);
    }

    std::cout << "X_train shape: " << X_train.rows() << "x" << X_train.cols() << std::endl;
    std::cout << "Y_train shape: " << Y_train.rows() << "x" << Y_train.cols() << std::endl;


    YANN::numeric_t learning_rate = 0.001f;
    size_t epochs = 1000;
    model.fit(X_train, Y_train, learning_rate, epochs);

    
    // Test model on 10000 pairs of numbers between -1 and 1
    size_t samples = 100;
    YANN::numeric_t range = 1.0f;
    YANN::matrix_t add_input = YANN::math_api::createMatrix(samples, 3);
    YANN::matrix_t add_output = YANN::math_api::createMatrix(samples, 1);
    
    for(size_t i = 0; i < samples; ++i)
    {
        add_input(i, 0) = (YANN::numeric_t)(2.0f * ((float)rand() / RAND_MAX) - 1.0f);
        add_input(i, 1) = (YANN::numeric_t)(2.0f * ((float)rand() / RAND_MAX) - 1.0f);
        add_input(i, 2) = 0; // 0 is the label for addition
    }

    for(size_t i = 0 ; i < samples ; i++)
    {
        add_output(i, 0) = YANN::math_api::matrixAt(model.forward(YANN::math_api::matrixTranspose(YANN::math_api::matrixRow(add_input, i))), 0, 0);
    }
    
    YANN::matrix_t add_results = YANN::math_api::createMatrix(samples, 3);


    add_results.col(0) = add_input.col(0);
    add_results.col(1) = add_input.col(1);
    add_results.col(2) = add_output.col(0);


    YANN::Utils::FileIO::saveMatrix(add_results, "add_results.txt");
    

    return 0;
}
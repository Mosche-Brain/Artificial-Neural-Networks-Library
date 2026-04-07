#define ENABLE_DEBUG_OUTPUTS
#include <iostream>
#include <YANN/Models/Sequential.hpp>
#include <YANN/Utility/FileIO.hpp>

// YANN::runtime_config::DEBUG_VERBOSITY = 3;

int main()
{
    YANN::math_api::setUsedThreadCount(8);
    size_t hidden_layer_size = 64;
    YANN::Models::Sequential model({
        YANN::Models::Layers::Input::createUnique(3),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(hidden_layer_size, "leaky_relu"),
        YANN::Models::Layers::Dense::createUnique(1, "linear")
    });

    YANN::matrix_t training_data;
    YANN::Utils::FileIO::readMatrix(training_data, "arithmetic.data");

    // normalize training data
    YANN::math_api::matrixNormalizeInPlace(training_data);

    std::cout << "Training data shape: " << training_data.rows() << "x" << training_data.cols() << std::endl;

    // std::cout << "Random samples from training data:" << std::endl;
    // for(size_t i = 0; i < 30; ++i)
    // {
    //     size_t index = rand() % YANN::math_api::matrixRows(training_data);
    //     std::cout << "X: " << training_data.row(index) << std::endl;
    // }

    // return 0;

    // YANN::matrix_t X_train = training_data.block(0, 0, training_data.rows(), 2);
    YANN::matrix_t X_train = YANN::math_api::createMatrix(training_data.rows(), 3);
    // YANN::matrix_t Y_train = training_data.block(0, 3, training_data.rows(), 3);
    YANN::matrix_t Y_train = YANN::math_api::createMatrix(training_data.rows(), 1);
    
    for(size_t i = 0; i < YANN::math_api::matrixRows(training_data); ++i)
    {
        for(size_t j = 0; j < 3; ++j)
        {
            X_train(i, j) = training_data(i, j);
        }
        Y_train(i, 0) = training_data(i, 3);
    }

    // print random 30 rows from X_train and Y_train
    std::cout << "Random samples from training data:" << std::endl;
    for(size_t i = 0; i < 30; ++i)
    {
        size_t index = rand() % YANN::math_api::matrixRows(X_train);
        std::cout << "X: " << X_train.row(index) << " Y: " << Y_train.row(index) << std::endl;
    }


    // return 0;
    std::cout << "X_train shape: " << YANN::math_api::matrixRows(X_train) << "x" << YANN::math_api::matrixCols(X_train) << std::endl;
    std::cout << "Y_train shape: " << YANN::math_api::matrixRows(Y_train) << "x" << YANN::math_api::matrixCols(Y_train) << std::endl;


    YANN::numeric_t learning_rate = 0.00001f;
    size_t epochs = 5000;
    model.fit(X_train, Y_train, learning_rate, epochs);
    
    size_t samples = 1000;
    YANN::numeric_t range = 100.0f;
    YANN::matrix_t arguments = YANN::math_api::createMatrix(samples, 2);
    YANN::matrix_t add_results = YANN::math_api::createMatrix(samples, 1);
    YANN::matrix_t subtract_results = YANN::math_api::createMatrix(samples, 1);
    YANN::matrix_t multiply_results = YANN::math_api::createMatrix(samples, 1);
    YANN::matrix_t divide_results = YANN::math_api::createMatrix(samples, 1);

    for(size_t i = 0; i < samples; ++i)
    {
        YANN::numeric_t a = (YANN::numeric_t)(range * ((float)rand() / RAND_MAX) - range / 2.0f);
        YANN::numeric_t b = (YANN::numeric_t)(range * ((float)rand() / RAND_MAX) - range / 2.0f);
        arguments(i, 0) = a;
        arguments(i, 1) = b;
        YANN::numeric_t add_label = 1, subtract_label = -1, multiply_label = 2, divide_label = -2;

        add_results(i, 0) = YANN::math_api::matrixAt(model.forward(YANN::math_api::createVector({a, b, add_label})), 0, 0);
        subtract_results(i, 0) = YANN::math_api::matrixAt(model.forward(YANN::math_api::createVector({a, b, subtract_label})), 0, 0);
        multiply_results(i, 0) = YANN::math_api::matrixAt(model.forward(YANN::math_api::createVector({a, b, multiply_label})), 0, 0);
        divide_results(i, 0) = YANN::math_api::matrixAt(model.forward(YANN::math_api::createVector({a, b, divide_label})), 0, 0);

    }

    YANN::matrix_t results = YANN::math_api::createMatrix(samples, 6);



    results.col(0) = arguments.col(0);
    results.col(1) = arguments.col(1);
    results.col(2) = add_results.col(0);
    results.col(3) = subtract_results.col(0);
    results.col(4) = multiply_results.col(0);
    results.col(5) = divide_results.col(0);     


    YANN::Utils::FileIO::saveMatrix(results, "results.txt");
    YANN::Utils::FileIO::saveSequentialModel(model, "model_params.bson");

    return 0;
}
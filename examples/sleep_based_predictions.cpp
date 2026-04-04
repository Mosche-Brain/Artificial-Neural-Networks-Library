#include <YANN/Models/Sequential.hpp>

// #include "lazycsv/lazycsv.hpp"

int main()
{
    YANN::Models::Sequential model
    ({
        YANN::Models::Layers::Input::createUnique(1),
        YANN::Models::Layers::Dense::createUnique(1024, "tanh"),
        YANN::Models::Layers::Dense::createUnique(1, "sigmoid"),
    });

    size_t num_samples = 84;
    YANN::vector_t X_train = YANN::vector_t::LinSpaced(num_samples, static_cast<YANN::numeric_t>(-6 * M_PI), static_cast<YANN::numeric_t>(6 * M_PI));
    
    YANN::vector_t Y_train(num_samples);
    for(size_t i = 0; i < num_samples; ++i)
    {
        Y_train(i) = std::sin(X_train(i));
    }

    YANN::vector_t X_eval = YANN::vector_t::LinSpaced(1024, static_cast<YANN::numeric_t>(-8 * M_PI), static_cast<YANN::numeric_t>(8 * M_PI));

    
    size_t epochs = 1000;
    YANN::numeric_t learning_rate = (YANN::numeric_t)0.0001f;
    
    YANN::vector_t Y_model_a_pretrain(YANN::math_api::vectorSize(X_eval));
    for(size_t i = 0 ; i < YANN::math_api::vectorSize(X_eval) ; i++)
    {
        Y_model_a_pretrain[i] = model.forward(YANN::math_api::createMatrix({{X_eval[i]}}))(0, 0);
    }
    
    model.fit(X_train, Y_train, learning_rate, epochs);

    return 0;
}
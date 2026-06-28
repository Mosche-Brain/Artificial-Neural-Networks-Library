#define ENABLE_DEBUG_OUTPUT
#include <iostream>


//#define CUM_USE_F32
#include <cum/Core.hpp>
#include <cum/cum.hpp>
#include <cum/Matrix.hpp>
#include <cum/random.hpp>

#include <YANN/Models/Sequential.hpp>
#include <YANN/Utility/FileIO.hpp>
#include <YANN/Utility/logs.hpp>
#include <YANN/runtime_config.hpp>

int main()
{
    cum::cum(cum::CUM_DEVICE::GPU);
    std::cout << "sizeof cumeric_t in bytes: " << sizeof(cum::cumeric_t) << '\n';
    YANN::runtime_config::set_verbosity(4);

    YANN::Models::Sequential sequential({
        YANN::Models::Layers::Input::createUnique(2),
        YANN::Models::Layers::Dense::createUnique(3, "sigmoid"),
        YANN::Models::Layers::Dense::createUnique(1, "sigmoid")
    });

    cum::Matrix x_train(4, 2,
                       {0, 0,
                        0, 1,
                        1, 0,
                        1, 1});

    cum::Matrix y_train(4, 1, {0, 1, 1, 0});
    cum::Matrix y_eval(4, 1);

    cum::cumeric_t rate = 0.1_c;
    size_t epochs = 1;

    for(int i = 0 ; i < 4 ; i++)
        y_eval(i, 0) = sequential.forward(x_train.row(i))(0,0);

    std::cout << "pretrain:\n";
    std::cout << '[' << x_train(0, 0) << ',' << x_train(0, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 0) << "\n";
    std::cout << '[' << x_train(1, 0) << ',' << x_train(1, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 1) << "\n";
    std::cout << '[' << x_train(2, 0) << ',' << x_train(2, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 2) << "\n";
    std::cout << '[' << x_train(3, 0) << ',' << x_train(3, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 3) << "\n";

    sequential.setLossFunction(YANN::Utils::loss::LossFunction::binary_cross_entropy);

    sequential.fit(x_train, y_train, rate, epochs);

    // for(int i = 0 ; i < sequential.getLayersCount() ; i++)
    // {
    //     std::cout << "Layer " << i << "\n";
    //     std::cout << "\tweights dimensions " << YANN::Utils::logs::show_matrix_dimensions(sequential.getWeights(i)) << "\n";
    //     std::cout << "\tbiases dimensions " << YANN::Utils::logs::show_matrix_dimensions(sequential.getBiases(i)) << "\n";
    // }

    cum::Matrix sample(1, 2, 1._c);


    for(int i = 0 ; i < 4 ; i++)
        y_eval(i, 0) = sequential.forward(x_train.row(i))(0,0);

    std::cout << "aftertrain:\n";
    std::cout << '[' << x_train(0, 0) << ',' << x_train(0, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 0) << "\n";
    std::cout << '[' << x_train(1, 0) << ',' << x_train(1, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 1) << "\n";
    std::cout << '[' << x_train(2, 0) << ',' << x_train(2, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 2) << "\n";
    std::cout << '[' << x_train(3, 0) << ',' << x_train(3, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 3) << "\n";

    cum::decum();

    return 0;
}

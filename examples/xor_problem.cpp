#include <iostream>

#include <cum/Core.hpp>
#include <cum/cum.hpp>
#include <cum/Matrix.hpp>

#include <YANN/Models/Sequential.hpp>
#include <YANN/Utility/FileIO.hpp>
#include <YANN/runtime_config.hpp>

int main()
{
    cum::cum(cum::CUM_DEVICE::CPU);
    std::cout << "sizeof cumeric_t in bytes: " << sizeof(cum::cumeric_t) << '\n';
    YANN::runtime_config::set_verbosity(0);

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
    size_t epochs = 4000;

    for(int i = 0 ; i < 4 ; i++)
        y_eval(i, 0) = sequential.forward(x_train.row(i).transpose())(0,0);

    std::cout << "pretrain:\n";
    std::cout << '[' << x_train(0, 0) << ',' << x_train(0, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 0) << "\n";
    std::cout << '[' << x_train(1, 0) << ',' << x_train(1, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 1) << "\n";
    std::cout << '[' << x_train(2, 0) << ',' << x_train(2, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 2) << "\n";
    std::cout << '[' << x_train(3, 0) << ',' << x_train(3, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 3) << "\n";

    sequential.setLossFunction(YANN::Utils::loss::LossFunction::binary_cross_entropy);

    sequential.fit(x_train, y_train, rate, epochs);

    cum::Matrix sample(1, 2, 1._c);


    for(int i = 0 ; i < 4 ; i++)
        y_eval(i, 0) = sequential.forward(x_train.row(i).transpose())(0,0);

    std::cout << "aftertrain:\n";
    std::cout << '[' << x_train(0, 0) << ',' << x_train(0, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 0) << "\n";
    std::cout << '[' << x_train(1, 0) << ',' << x_train(1, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 1) << "\n";
    std::cout << '[' << x_train(2, 0) << ',' << x_train(2, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 2) << "\n";
    std::cout << '[' << x_train(3, 0) << ',' << x_train(3, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 3) << "\n";

    cum::decum();

    return 0;
}

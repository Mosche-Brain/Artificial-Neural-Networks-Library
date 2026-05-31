#define ENABLE_DEBUG_OUTPUT
#include <iostream>


#define CUM_USE_F16
#define BUILD_USE_MKL
#include <cum/Core.hpp>
#include <cum/cum.hpp>
#include <cum/Matrix.hpp>

#include <YANN/Models/Sequential.hpp>
#include <YANN/Utility/FileIO.hpp>

#include <YANN/Utility/logs.hpp>

#include <cum/random.hpp>

int main()
{
    cum::cum(cum::CUM_DEVICE::GPU);

    std::cout << "sizeof cumeric_t in bytes: " << sizeof(cum::cumeric_t) << '\n';

    std::cout << "YANN Playground\n";

    // size_t N = 8;
    // cum::cumeric_t* buffer = (cum::cumeric_t*)malloc(N * sizeof(cum::cumeric_t));

    // std::cout << YANN::Utils::logs::matrixToString(A);
    // return 0;
    // cum::random::uniform(A.data(), N, -1._c, 1._c);

    // return 0;

    YANN::Models::Sequential sequential({
        YANN::Models::Layers::Input::createUnique(2),
        YANN::Models::Layers::Dense::createUnique(3, "sigmoid"),
        YANN::Models::Layers::Dense::createUnique(1, "sigmoid")
    });

    std::cout << "Model initialized\n";

    cum::Matrix x_train(4, 2,
                       {0, 0,
                        0, 1,
                        1, 0,
                        1, 1});

    cum::Matrix y_train(4, 1, {0, 1, 1, 0});


    cum::cumeric_t rate = 0.01_c;
    size_t epochs = 50;

    sequential.fit(x_train, y_train, rate, epochs);

    // for(int i = 0 ; i < sequential.getLayersCount() ; i++)
    // {
    //     std::cout << "Layer " << i << "\n";
    //     std::cout << "\tweights dimensions " << YANN::Utils::logs::show_matrix_dimensions(sequential.getWeights(i)) << "\n";
    //     std::cout << "\tbiases dimensions " << YANN::Utils::logs::show_matrix_dimensions(sequential.getBiases(i)) << "\n";
    // }

    cum::Matrix sample(1, 2, 1._c);
    std::cout << "Sample created\n"; 

    // forward sample 1024 times
    // for(int i = 0 ; i < 1024 ; i++)
    // {
        // sequential.forward(sample);
    // }


    // cum::Matrix result = sequential.forward(sample);
    std::cout << "Forward pass completed\n";

    // std::cout << YANN::Utils::logs::matrixToString(result);


    std::cout << "end\n";
    cum::decum();

    return 0;
}

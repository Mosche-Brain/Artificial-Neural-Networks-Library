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

int main()
{
    cum::cum(cum::CUM_DEVICE::CPU);

    std::cout << "sizeof cumeric_t in bytes: " << sizeof(cum::cumeric_t) << '\n';

    std::cout << "YANN Playground\n";

    YANN::Models::Sequential sequential({
        YANN::Models::Layers::Input::createUnique(4),
        YANN::Models::Layers::Dense::createUnique(4, "relu"),
        YANN::Models::Layers::Dense::createUnique(4, "sigmoid")
    });

    std::cout << "Model initialized\n";

    for(int i = 0 ; i < sequential.getLayersCount() ; i++)
    {
        std::cout << "Layer " << i << "\n";
        std::cout << "\tweights dimensions " << YANN::Utils::logs::show_matrix_dimensions(sequential.getWeights(i)) << "\n";
        std::cout << "\tbiases dimensions " << YANN::Utils::logs::show_matrix_dimensions(sequential.getBiases(i)) << "\n";
    }

    cum::Matrix sample(1, 4, 1._c);
    std::cout << "Sample created\n";
    sequential.forward(sample);
    // cum::Matrix result = sequential.forward(sample);
    std::cout << "Forward pass completed\n";

    // model.addLayer(YANN::Models::Layers::Input::createUnique(2));
    // model.addLayer(YANN::Models::Layers::Dense::createUnique(4, "relu"));
    // model.addLayer(YANN::Models::Layers::Dense::createUnique(1, "sigmoid"));

    cum::decum();
    return 0;
}

    // Foo fo(4);

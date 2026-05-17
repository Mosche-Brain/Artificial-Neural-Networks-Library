#define ENABLE_DEBUG_OUTPUTS
#include <iostream>
#include <YANN/Models/Sequential.hpp>
#include <YANN/Utility/FileIO.hpp>

#define CUM_USE_F16
#include <cum/cum.hpp>

int main()
{
    cum::cum(cum::CUM_DEVICE::GPU);

    YANN::Models::Sequential model({
        YANN::Models::Layers::Input::createUnique(2),
        YANN::Models::Layers::Dense::createUnique(4, "relu"),
        YANN::Models::Layers::Dense::createUnique(1, "sigmoid")
    });
    // model.addLayer(YANN::Models::Layers::Input::createUnique(2));
    // model.addLayer(YANN::Models::Layers::Dense::createUnique(4, "relu"));
    // model.addLayer(YANN::Models::Layers::Dense::createUnique(1, "sigmoid"));
    return 0;
    
    cum::Matrix train_x(4, 2);
    train_x(0, 0) = 0.0f; train_x(0, 1) = 0.0f;
    train_x(1, 0) = 0.0f; train_x(1, 1) = 1.0f;
    train_x(2, 0) = 1.0f; train_x(2, 1) = 0.0f;
    train_x(3, 0) = 1.0f; train_x(3, 1) = 1.0f;

    cum::Vector train_y(4);
    train_y[0] = 0.0f;
    train_y[1] = 1.0f;
    train_y[2] = 1.0f;
    train_y[3] = 0.0f;

    model.fit(train_x, train_y, 0.05f, 10000);

    cum::decum();
    return 0;
}

// #define fsycl
#include <yann/models/Perceptron.hpp>
#include <cum/cum.hpp>

int main()
{
    cum::cum(cum::CUM_DEVICE::GPU);
    yann::models::Perceptron p(2, "sigmoid");

    cum::Matrix train_x(4, 2);
    train_x(0, 0) = 0.0f; train_x(0, 1) = 0.0f;
    train_x(1, 0) = 0.0f; train_x(1, 1) = 1.0f;
    train_x(2, 0) = 1.0f; train_x(2, 1) = 0.0f;
    train_x(3, 0) = 1.0f; train_x(3, 1) = 1.0f;

    cum::Vector train_y(4);
    train_y[0] = 0.0f;
    train_y[1] = 1.0f;
    train_y[2] = 1.0f;
    train_y[3] = 1.0f;

    p.fit(train_x, train_y, 0.05f, 1000);

    cum::Vector input(2);
    input[0] = 0.0f;
    input[1] = 1.0f;

    cum::cumeric_t output = p.predict(input);

    cum::decum();
    return 0;
}
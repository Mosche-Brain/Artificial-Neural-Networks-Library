#define ENABLE_DEBUG_OUTPUT
#include <iostream>
#include <YANN/Models/Sequential.hpp>
#include <YANN/Utility/FileIO.hpp>

#define CUM_USE_F16
#define BUILD_USE_MKL
#include <cum/Core.hpp>
#include <cum/cum.hpp>
#include <cum/Matrix.hpp>


class Foo
{
public:
    Foo(const size_t size)
    {
        mat = cum::Matrix(1, size, 0._c);
    }
private:
    cum::Matrix mat;
};


int main()
{
    cum::cum(cum::CUM_DEVICE::CPU);

    std::cout << "sizeof cumeric_t in bytes: " << sizeof(cum::cumeric_t) << '\n';

    // cum::Matrix A(2, 2, 1.0_c);
    // cum::Matrix A;

    // A = cum::Matrix(2, 2, 1._c);

    

    // display content of matrix A
    // std::cout << "Matrix A:\n";
    // for (int i = 0; i < A.rows(); ++i) 
    // {
    //     for (int j = 0; j < A.cols(); ++j) 
    //     {
    //         std::cout << A(i, j) << "\t";
    //     }
    //     std::cout << '\n';
    // }

    // Foo fo(4);

    std::cout << "YANN Playground\n";

    YANN::Models::Sequential sequential({
        YANN::Models::Layers::Input::createUnique(4),
        YANN::Models::Layers::Dense::createUnique(4, "relu"),
        YANN::Models::Layers::Dense::createUnique(1, "sigmoid")
    });
    // model.addLayer(YANN::Models::Layers::Input::createUnique(2));
    // model.addLayer(YANN::Models::Layers::Dense::createUnique(4, "relu"));
    // model.addLayer(YANN::Models::Layers::Dense::createUnique(1, "sigmoid"));

    cum::decum();
    return 0;
}

    // Foo fo(4);

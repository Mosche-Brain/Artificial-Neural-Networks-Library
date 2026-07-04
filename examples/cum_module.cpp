#include <cum/Matrix.hpp>
#include <cum/cum.hpp>

#include <YANN/Utility/logs.hpp>
#include <iostream>

int main()
{
    cum::cum(cum::CUM_DEVICE::GPU);

    cum::Matrix mat(2, 2, {1, 2, 3, 4});

    std::cout << "before transpose:\n" << yann::Utils::logs::matrixToString(mat);

    mat.transposeInPlace();

    std::cout << "after transpose:\n" << yann::Utils::logs::matrixToString(mat);

    cum::decum();

    return 0;
}
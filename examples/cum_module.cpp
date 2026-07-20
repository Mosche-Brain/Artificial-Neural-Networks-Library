#include <cum/Matrix.hpp>
#include <cum/cum.hpp>

#include <yann/utils/logs.hpp>
#include <iostream>

int main()
{
    cum::cum(cum::CUM_DEVICE::GPU);

    cum::Matrix mat(2, 2, {1, 2, 3, 4});

    std::cout << "before transpose:\n" << yann::utils::logs::matrixToString(mat);

    mat.transposeInPlace();

    std::cout << "after transpose:\n" << yann::utils::logs::matrixToString(mat);

    cum::decum();

    return 0;
}
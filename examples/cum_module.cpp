#include <cum/Tensor.hpp>
#include <cum/Vector.hpp>
#include <cum/Matrix.hpp>
#include <cum/cum.hpp>

#include <iostream>

import yann;

int main()
{
    cum::cum(cum::DEVICE::GPU);

    cum::Matrix mat(2, 2, {1, 2, 3, 4});

    std::cout << "before transpose:\n" << yann::utils::formating::matrixToString(mat);

    mat.transposeInPlace();

    std::cout << "after transpose:\n" << yann::utils::formating::matrixToString(mat);

    cum::decum();

    return 0;
}
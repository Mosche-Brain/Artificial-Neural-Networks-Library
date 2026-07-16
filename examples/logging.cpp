//
// Created by jaro on 6/25/26.
//

#include <YANN/Utility/logs.hpp>

#include <cum/Matrix.hpp>

int main()
{
    cum::Matrix A(3, 3, {0.0002, 2.2, 3, 4, 5, 6, 7, 8, 9});
    cum::Matrix B = A.transpose();

    std::cout << yann::utils::logs::matricesWithArrowToString(A, B, 0, 16);

    return 0;
}
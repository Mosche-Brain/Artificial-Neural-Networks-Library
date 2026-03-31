#include <iostream>
#include <YANN/math_api.hpp>

int main()
{
    YANN::math_api::matrix_t A = YANN::math_api::createMatrix({
        {1.0f, 2.0f},
        {3.0f, 4.0f},
        {5.0f, 6.0f}
    });

    YANN::math_api::matrix_t B = YANN::math_api::createMatrix({
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
    });

    // YANN::math_api::matrix_t C = YANN::math_api::matrixAdd(A, B);
    YANN::math_api::matrix_t C = YANN::math_api::matrixMultiply(A, B);

    std::cout << "Matrix A:\n" << A << "\n\n";
    std::cout << "Matrix B:\n" << B << "\n\n";
    std::cout << "Matrix C:\n" << C << "\n\n";

    // Eigen::Matrix<float, 3, 3> eigenA = A;
    // A << 1.0f, 2.0f, 3.0f,
    //      4.0f, 5.0f, 6.0f,
    //      7.0f, 8.0f, 9.0f;

    // Eigen::Matrix<float, 3, 3> eigenB = B;
    // B << 9.0f, 8.0f, 7.0f,
    //      6.0f, 5.0f, 4.0f,
    //      3.0f, 2.0f, 1.0f;

    // Eigen::Matrix<float, 3, 3> eigenC = eigenA + eigenB;

    // std::cout << eigenC << "\n\n";

    // YANN::math_api::matrix_t colwiseSum = YANN::math_api::matrixColwiseSum(A);
    // std::cout << colwiseSum << '\n';

    return 0;
}
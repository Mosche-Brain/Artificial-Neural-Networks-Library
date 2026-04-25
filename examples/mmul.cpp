#include <YANN/math_api.hpp>

int main()
{

    // #pragma intel optimization_parameter
    // multiplt two 1024x1024 matrices 1024 times and measure the time taken
    size_t matrix_size = 1024;
    YANN::matrix_t A = YANN::math_api::createMatrix(matrix_size, matrix_size, 1.0f);
    YANN::matrix_t B = YANN::math_api::createMatrix(matrix_size, matrix_size, 2.0f);
    YANN::matrix_t C = YANN::math_api::createMatrix(matrix_size, matrix_size);


    for(int i = 0 ; i < 1024 ; i++)
    {
        C = A * B;
    }

    return 0;
}
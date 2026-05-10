#include "cum/cum.hpp"
#include "cum/Core.hpp"
#include "cum/LinearAlgbebra.hpp"
#include "cum/Matrix.hpp"

#include <cstdlib>
#include <iostream>

#define fsycl
#include <sycl/sycl.hpp>

int main()
{
    cum::cum();

    // size_t N = 8;
    // // cum::cumeric_t v[] = {1, 2.3, 9.4, 1.7, 6.7, 6.9, 9.6, 7.6};
    // // cum::cumeric_t u[] = {6.7, 6.9, 9.6, 7.6, 1, 2.3, 9.4, 1.7};
    // // cum::cumeric_t* r = (cum::cumeric_t*)malloc(N * sizeof(cum::cumeric_t));
    // cum::cumeric_t v[] = {1, 2.3, 9.4, 1.7, 6.7, 6.9, 9.6, 7.6};
    // cum::cumeric_t u[] = {6.7, 6.9, 9.6, 7.6, 1, 2.3, 9.4, 1.7};
    // cum::cumeric_t* r = (cum::cumeric_t*)calloc(N, sizeof(cum::cumeric_t));

    // // cum::LinearAlgebra::add(r, v, u, N);
    // cum::LinearAlgebra::addInPlace(v, u, N);

    // for(int i = 0 ; i < N ; i++)
    // {
    //     std::cout << v[i] << ' ';
    // }
    // std::cout << '\n';
    

    // size_t n = 2, m = 2, k = 2;
    // cum::cumeric_t A[] = {1, 2,
    //                       3, 4};

    // cum::cumeric_t B[] = {6, 9,
    //                       6, 7};

    // cum::cumeric_t* C = (cum::cumeric_t*)calloc(m * n, sizeof(cum::cumeric_t));

    // cum::LinearAlgebra::matMul(C, A, B, n, m, k);

    // for(size_t i = 0 ; i < m; i++)
    // {
    //     for (size_t j = 0; j < n; j++)
    //     {
    //         std::cout << C[i * n + j] << ' ';
    //     }
    //     std::cout << '\n';
    // }

    // free(r);
    // free(C);

    cum::Matrix A(8, 8, 6.0);
    cum::Matrix B(8, 8, 9.0);

    cum::Matrix C = A * B;

        for(size_t i = 0 ; i < C.rows(); i++)
    {
        for (size_t j = 0; j < C.cols(); j++)
        {
            std::cout << C.at(i, j) << '\t';
        }
        std::cout << '\n';
    }

    cum::decum();

    return 0;
}
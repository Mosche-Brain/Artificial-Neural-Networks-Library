#include "cum/cum.hpp"
#include "cum/Core.hpp"
#include "cum/LinearAlgebra.hpp"
#include "cum/Matrix.hpp"

#include <cstdlib>
#include <iostream>

#define fsycl
#include <sycl/sycl.hpp>

#include <ctime>
#include <unistd.h>

bool is_addition_correct()
{
    cum::Matrix A(2, 2, 1.0);
    cum::Matrix B(2, 2, 2.0);

    cum::Matrix C = A + B;

    for(size_t i = 0 ; i < C.rows() ; i++)
        for(size_t j = 0 ; j < C.cols() ; j++)
            if(C.at(i, j) != 3.0)
                return false;
    
    return true;
}

double benchmark()
{
    cum::Matrix A(4096, 4096, 6.0);
    cum::Matrix B(4096, 4096, 9.0);

    //measure time of 128 multiplications
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0 ; i < 256 ; i++)
    {
        A * B;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    long long elapsed = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
    double time = elapsed / 1000000000.0;
    return time;
}

int main()
{
    cum::cum(cum::DEVICE::GPU);


    std::cout << "Testing addition correctness... ";
    if(is_addition_correct())
        std::cout << "PASSED" << std::endl;
    else
        std::cout << "FAILED" << std::endl;
    sleep(1);

    std::cout << "Time taken on GPU: ";
    double gpu_time = benchmark();
    std::cout << gpu_time << " seconds" << std::endl;

    cum::recum(cum::DEVICE::CPU);
    sleep(5);

    std::cout << "Time taken on CPU: ";
    double cpu_time = benchmark();
    std::cout << cpu_time << " seconds" << std::endl;
    cum::decum();

    return 0;
}
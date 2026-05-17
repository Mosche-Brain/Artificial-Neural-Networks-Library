#include <cum/Matrix.hpp>
#include <cum/cum.hpp>

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
    cum::cum(cum::CUM_DEVICE::GPU);

    std::cout << "Time taken on GPU: ";
    double gpu_time = benchmark();
    std::cout << gpu_time << " seconds" << std::endl;

    cum::recum(cum::CUM_DEVICE::CPU);

    std::cout << "Time taken on CPU: ";
    double cpu_time = benchmark();
    std::cout << cpu_time << " seconds" << std::endl;
    cum::decum();

    return 0;
}
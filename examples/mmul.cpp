#include <cum/Matrix.hpp>
#include <cum/cum.hpp>

size_t EDGE_SIZE=2048;
// size_t EDGE_SIZE=4096;
// size_t EDGE_SIZE=8192;

double benchmark()
{
    cum::Matrix A(EDGE_SIZE, EDGE_SIZE, 6.0);
    cum::Matrix B(EDGE_SIZE, EDGE_SIZE, 9.0);

    //measure time of 128 multiplications
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0 ; i < 1028*32 ; i++)
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

    // cum::recum(cum::CUM_DEVICE::CPU);

    // std::cout << "Time taken on CPU: ";
    // double cpu_time = benchmark();
    // std::cout << cpu_time << " seconds" << std::endl;
    // cum::decum();

    return 0;
}

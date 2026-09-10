#include <cum/Matrix.hpp>
#include <cum/cum.hpp>

#include <cum/memory.hpp>

#include <vector>

// size_t EDGE_SIZE=2048;
size_t EDGE_SIZE=4096;
// size_t EDGE_SIZE=8192;

double benchmark()
{
    // cum::Matrix A(EDGE_SIZE, EDGE_SIZE, 6.0);
    // cum::Matrix B(EDGE_SIZE, EDGE_SIZE, 9.0);

    // cum::memory::prefetch(A.data(), EDGE_SIZE * EDGE_SIZE);
    // cum::memory::prefetch(B.data(), EDGE_SIZE * EDGE_SIZE);

    std::vector<cum::Matrix> mats1;
    std::vector<cum::Matrix> mats2;

    std::size_t count = 128;
    for (std::size_t i = 0; i < count; i++)
    {
        mats1.push_back(cum::Matrix(EDGE_SIZE, EDGE_SIZE, 6.0));
        mats2.push_back(cum::Matrix(EDGE_SIZE, EDGE_SIZE, 9.0));
    }

    for (std::size_t i = 0; i < count; i++)
    {
        cum::memory::prefetch(mats1[i].data(), EDGE_SIZE * EDGE_SIZE);
        cum::memory::prefetch(mats2[i].data(), EDGE_SIZE * EDGE_SIZE);
    }


    //measure time of 128 multiplications
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0 ; i < count ; i++)
    {
        mats1[i] * mats2[i];
        // A * B;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    long long elapsed = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
    double time = elapsed / 1000000000.0;
    return time;
}

int main()
{
    cum::cum(cum::DEVICE::GPU);

    std::cout << "Time taken on GPU: ";
    double gpu_time = benchmark();
    std::cout << gpu_time << " seconds" << std::endl;

    // cum::recum(cum::DEVICE::CPU);

    // std::cout << "Time taken on CPU: ";
    // double cpu_time = benchmark();
    // std::cout << cpu_time << " seconds" << std::endl;
    // cum::decum();

    return 0;
}

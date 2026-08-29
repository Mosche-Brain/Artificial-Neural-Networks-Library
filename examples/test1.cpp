#include <iostream>
#include <cum/Matrix.hpp>

using namespace cum;
void testMul(std::size_t k)
{
    std::cout << "Testing 1x" << k
              << " * "
              << k << "x1" << std::endl;

    Matrix a = Matrix::Ones(1, k);
    Matrix b = Matrix::Ones(k, 1);

    Matrix c = a * b;

    std::cout << "OK" << std::endl;
}

void test1()
{
    Matrix input(1, 3, {
        1_c, 2_c, 3_c
    });

    Matrix weights(3, 1, {
        0.1_c,
        0.2_c,
        0.3_c
    });

    Matrix output = input * weights;

    std::cout << "forward passed" << std::endl;
}


void test3()
{
    for(int i=0;i<10000;i++)
    {
        Matrix a = Matrix::Random(1,3);
        Matrix b = Matrix::Random(3,1);

        Matrix c = a * b;
    }
}

void test2()
{
    for(std::size_t k = 1; k <= 32; ++k)
    {
        try
        {
            Matrix a = Matrix::Ones(1, k);
            Matrix b = Matrix::Ones(k, 1);

            Matrix c = a * b;

            std::cout << "[OK] k=" << k << '\n';
        }
        catch(...)
        {
            std::cout << "[FAIL] k=" << k << '\n';
        }
    }
}

int main()
{
    // testMul(2);
    // testMul(3);
    // testMul(4);
    // testMul(5);
    // testMul(6);
    // testMul(7);

    // test1();
    // test2();

    test3();

    return 0;
}
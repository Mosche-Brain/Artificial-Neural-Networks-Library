#include "Perceptron.hpp"

namespace SNN::Models
{
    Perceptron::Perceptron(int inputLenght, std::function<float_t(float_t)> activation)
    {

    }

    


    void Perceptron::init()
    {
        this->weight = Eigen::VectorXf::Random(_inputWidth_);
    }
}
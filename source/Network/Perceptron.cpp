#include "Perceptron.hpp"

Perceptron::Perceptron(int32_t inputs)
{
    this->weights = VectorXd::Random(inputs);
    this->bias = 0;
}

double Perceptron::forward(VectorXd x)
{
    if(x.size() != this->weights.size())
    {
        throw std::invalid_argument("Input size does not match the number of weights");
    }

    double weighted_sum = x.dot(this->weights) + bias;

    return this->activation_function(weighted_sum);
}

double Perceptron::activation_function(double x)
{
    return 1 / (1 + exp(-x));
}

void Perceptron::train(MatrixXd x_train, VectorXd expected, int n_iter, double learning_rate)
{
    if(x_train.cols() != this->weights.size()) 
    {
        throw std::invalid_argument("Input size does not match the number of weights");
    }

    int input_n = x_train.size();

    for(int i = 0 ; i < n_iter ; i++)
    {
        for(int j = 0 ; j < x_train.rows() ; j++)
        {
            double y = this->forward(x_train.row(j));
            double misc = expected[j] - y;
            double correction = misc * learning_rate;

            for(int k = 0 ; k < weights.size() ; k++)
            {
                this->weights[k] += correction * x_train(j, k);
            }

            this->bias += correction;
        }
    }
}

std::vector<double> Perceptron::predict(MatrixXd x)
{
    std::vector<double> predictions;
    for(int i = 0 ; i < x.rows() ; i++)
    {
        double out = this->forward(x.row(i));
        predictions.push_back(out);
    }

    return predictions;
}

// void Perceptron::forward_signal(VectorXd x)
// {
//     if(connections.empty())
//         return;

//     //std::vector<double> predictions = this->forward(x);
//     double predictions = this->forward(x);

//     for(int i = 0 ; i < this->connections.size() ; i++)
//     {
//         //connections.at(i)->forward_signal(predictions);
//         connections.at(i)->
//     }
// }
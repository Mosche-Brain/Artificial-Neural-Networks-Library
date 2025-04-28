#include "Network/Layer.hpp"

/*
 * Wektorem wag dla każdego neuronu [i] jest wiersz [i] macierzy wag.
*/

Layer::Layer(int layer_size, int input_size, std::function<double(double)> func, std::function<double(double)> derivative, bool passive_layer)
{
    if(passive_layer)
        this->weights = MatrixXd::Ones(layer_size, input_size);
    else
        this->weights = MatrixXd::Random(layer_size, input_size) * 0.1;

    this->biases             = VectorXd::Zero(layer_size);
    this->outputs            = VectorXd::Zero(layer_size);
    this->delta              = VectorXd::Zero(layer_size);
    this->outputs_raw        = VectorXd::Zero(layer_size);
    this->derivative_outputs = VectorXd::Zero(layer_size);

    this->layer_size = layer_size;
    this->input_size = input_size;

    this->activation_function = func;
    this->activation_derivative = derivative;
}

Layer::Layer(int layer_size, int input_size, const char* func, bool passive_layer)
{
    // std::map<std::string, std::function<double(double)>> funcions =
    // {
    //     {"linear", pass},
    //     {"sigmoid", sigmoid},
    //     {"relu", RELu},
    //     {"tanh", tanh}
    // };

    /* Not yet finished */
}

VectorXd Layer::forward(VectorXd x, bool derivatives)
{
    input = x;

    if(x.size() != weights.cols())
    {
        std::cout << "input size doesn't match with weights\n";
    }

    // if (x.cols() == this->input_size && x.rows() == 1)
    // {
    //     output_vector = (this->weights * x.transpose()).col(0) + this->biases;
    // }
    // else if (x.cols() == this->input_size && x.rows() > 1)
    // {
    //     MatrixXd result = (this->weights * x.transpose()).colwise() + this->biases;
    //     output_vector = result.rowwise().mean(); 
    // }
    // else
    // {
    //     std::cerr << "Invalid input size for layer forward. Input: " 
    //               << x.rows() << "x" << x.cols() 
    //               << ", expected cols: " << this->input_size << "\n";
    //     return VectorXd::Zero(this->layer_size);
    // }    

    for (int i = 0; i < outputs.size(); ++i)
    {
        outputs_raw[i] = x.dot(weights.row(i).transpose()) + biases[i];
        
        if(derivatives)
        {
            derivative_outputs[i] = activation_derivative(outputs_raw[i]);
        }
        else
        {
            outputs[i] = activation_function(outputs_raw[i]);
        }
    }

    if(derivatives)
    {
        return derivative_outputs;
    }
    else
    {
        return outputs;
    }
}

VectorXd Layer::compute_delta(VectorXd target, bool output_layer, const Layer* next)
{
    VectorXd δ(layer_size);

    if(output_layer)
    {
        δ = (outputs - target).cwiseProduct(derivative_outputs);
    }
    else
    {
        // δ = (next->weights.transpose() * target).cwiseProduct(derivative_outputs);
        δ = (next->weights.transpose() * target).cwiseProduct(derivative_outputs);
    }

    this->delta = δ;

    return δ;
}

// VectorXd Layer::backprop(VectorXd δ, double rate)
// {
//     // weights_grad = derivative_outputs.dot(input.transpose());
//     weights_grad = δ * input.transpose();
//     biases_grad = δ;

//     std::cout << "grads\n";
//     std::cout << "delta " << δ <<'\n';
//     std::cout << "weigths " << weights <<'\n';

//     if(δ.size() == layer_size)
//     {
//         throw std::invalid_argument("invalid δ size");
//     }

//     VectorXd input_grad = weights.transpose() * δ;
    
//     std::cout << "weights\n";

//     this->weights -= weights_grad * rate;
//     this->biases -= biases_grad * rate;

//     return input_grad;
// }

void Layer::train(MatrixXd data, VectorXd expected, int n, double rate) /* basic training algorithm for one-layer networks */
{
    if(data.cols() != this->input_size)
    {
        std::cerr << "At training: Matrix size doesn`t match witch weights\n";
        return;
    }

    VectorXd results = VectorXd::Zero(this->outputs.size());
 
    for(int _ = 0 ; _ < n ; _++)
    {
        for(int rowIndex = 0 ; rowIndex < data.rows() ; rowIndex++)
        {
            //results = this->forward(data.row(rowIndex)).transpose();
            for(int neuronIndex = 0 ; neuronIndex < this->layer_size ; neuronIndex++)
            {
                VectorXd input = data.row(rowIndex).transpose();
                double target = expected[rowIndex];
                
                results = this->forward(data.row(rowIndex)).transpose();
                double prediction = results[neuronIndex];

                double error = target - results[neuronIndex];
                //double L = results[neuronIndex] * (1 - results[neuronIndex]);
            
                double correction = error * rate;
            
                for(int k = 0 ; k < this->weights.row(neuronIndex).size() ; k++)
                {
                    this->weights.row(neuronIndex)[k] += correction * data.row(rowIndex)[k];
                }

                //this->weights.row(i) += (correction * data.row(j));
                this->biases[neuronIndex] += correction;
            }

        }

    }
}
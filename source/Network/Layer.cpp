#include "Network/Layer.hpp"

Layer::Layer(int layer_size, int input_size, std::function<double(double)> func, bool passive_layer)
{
    if(passive_layer)
        this->weights = MatrixXd::Ones(layer_size, input_size);
    else
        this->weights = MatrixXd::Random(layer_size, input_size);

    this->biases  = VectorXd::Zero(layer_size);
    this->outputs = VectorXd::Zero(layer_size);

    this->layer_size = layer_size;
    this->input_size = input_size;

    this->activation_function = func;
}

VectorXd Layer::forward(MatrixXd input, bool override_output)
{
    VectorXd output_vector;

    if (input.cols() == this->input_size && input.rows() == 1)
    {
        output_vector = (this->weights * input.transpose()).col(0) + this->biases;
    }
    else if (input.cols() == this->input_size && input.rows() > 1)
    {
        MatrixXd result = (this->weights * input.transpose()).colwise() + this->biases;
        output_vector = result.rowwise().mean(); 
    }
    else
    {
        std::cerr << "Invalid input size for layer forward. Input: " 
                  << input.rows() << "x" << input.cols() 
                  << ", expected cols: " << this->input_size << "\n";
        return VectorXd::Zero(this->layer_size);
    }

    // Aktywacja
    for (int i = 0; i < output_vector.size(); ++i)
        output_vector[i] = this->activation_function(output_vector[i]);

    if (override_output)
        this->outputs = output_vector;

    return output_vector;
}

void Layer::train(MatrixXd data, VectorXd expected, int n, double rate)
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

VirtualNeuron Layer::neuron(int index)
{
    VirtualNeuron neuronObj;

    neuronObj.weights = &this->weights.row(index).data();
    neuronObj.bias    = &this->biases[index];

    return neuronObj;
}

std::vector<VirtualNeuron> Layer::getLayer()
{
    std::vector<VirtualNeuron> buff;
    for(int i = 0 ; i < this->layer_size ; i++)
    {
        buff.push_back(this->neuron(i));
    }

    return buff;
}
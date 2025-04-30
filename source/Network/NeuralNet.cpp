#include "Network/NeuralNet.hpp"

NeuralNet::NeuralNet(int layers_n, VectorXi layers_size)
{
    if(layers_n != layers_size.size())
        return;

    this->layers.push_back(new Layer(layers_size[0], 1, pass, pass_prim));
    //this->layers[0]->weights = MatrixXd::Ones();

    for(int i = 1 ; i < layers_n ; i++)
    {
        this->layers.push_back(new Layer(layers_size[i], layers_size[i - 1], sigmoid, sigmoid_prim));
    }



    // for(int i = 0 ; i < layers.size() ; i++)
    // {
    //     std::cout << "Layer size: " << layers[i]->layer_size << " Input size: " << layers[i]->input_size << '\n'; 
    // }
    
}

NeuralNet::NeuralNet(std::vector<Layer*> topology)
{
    this->layers = topology;
    this->layer_size = topology.size();
}

void NeuralNet::setLossFunction(std::function<double(VectorXd, VectorXd)> func)
{
    this->loss_function = func;
}

void NeuralNet::train(MatrixXd train_x, MatrixXd train_y, uint n_iter, float rate)
{
    for(int epoch = 0 ; epoch < n_iter ; epoch++)
    {
        std::cout << "Epoch " << epoch << " started\n";
        for(int i = 0 ; i < train_x.rows() ; i++)
        {
            std::cout << "Transposinging...\n";
            VectorXd x = train_x.row(i).transpose();
            VectorXd y = train_y.row(i).transpose();

            std::cout << "Forwarding...\n";
            VectorXd predicted = this->forward(x);
                                 this->forward(x, true);

            std::cout << "Compute loss...\n";
            std::cout << "Predicted:" << predicted.size() << "\n";
            std::cout << "target:" << y.size() << '\n';
            double loss = loss_function(predicted, y);
            
            std::cout << "Backprop...\n";
            this->backpropagate(y);
            this->update_weights(rate);

            std::cout << "loss " << loss << "\n";
        }
        std::cout << "Epoch " << epoch << " ended" << "\n";
    }

    std::cout << "Training ok\n";
}


VectorXd NeuralNet::forward(MatrixXd input, bool derivatives)
{
    // std::cout << "Size: " << input.cols() << '\n';
    // std::cout << "cols: " << input.cols() << '\n';
    // std::cout << "rows: " << input.rows() << '\n';

    if(input.rows() == 1 && input.cols() > 1)
        input.transposeInPlace();

    layers[0]->outputs = input;

    std::cout << "ok\n";
    for(int i = 1 ; i < this->layers.size() ; i++)
    {
        // std::cout << i << " Iteration\n";
        //input = layers[i]->outputs;
        
        //layers[i + 1]->outputs = layers[i + 1]->forward(layers[i]->outputs);
        //this->layers[i + 1]->forward(layers[i]->outputs.transpose());

        this->layers[i]->forward(layers[i - 1]->outputs, derivatives);
        // std::cout << "oki  size: " << this->layers[i]->outputs.size() << '\n';
    }
    
    return this->layers.back()->outputs;
}

MatrixXd NeuralNet::predict(MatrixXd input)
{
    MatrixXd output;

    for(int i = 0 ; i < input.rows() ; i++)
    {
        output << forward(input.row(i).transpose()).transpose();
    }

    return output;
}

/*VectorXd NeuralNet::forward(int currentIndex, int nextIndex)
{

    Layer* currentLayer = this->layers[currentIndex];
    Layer* nextLayer    = this->layers[nextIndex];

    VectorXd input = currentLayer->outputs;
    VectorXd output = nextLayer->forward(input);
 
    if(nextIndex >= this->layers.size() - 1)
        return output;


    return this->forward(currentIndex + 1, nextIndex + 1);
}*/

void NeuralNet::backpropagate(VectorXd expected)
{
    Layer* output_layer = layers.back();
    
    if(expected.size() != output_layer->outputs.size())
    {
        std::cout << "Output layer size doesn't match with train target\n";
        return;
    }

    //VectorXd network_error = (output_layer->outputs - expected) * output_layer->derivative_outputs;
    std::cout << "dudu\n";
    for(int j = 0 ; j < output_layer->layer_size ; j++)
    {
        // std::cout << "errors size:" << errors.size() << '\n';
        std::cout << "output size:" << output_layer->outputs.size() << '\n';
        std::cout << "target size:" << expected.size() << '\n';                
        // errors[j] = curr->outputs[j] - expected[j];
        double error = output_layer->outputs[j] - expected[j];
        output_layer->delta[j] = error * output_layer->derivative_outputs[j];
    }
   
    for(int i = layers.size() - 2 ; i > 0 ; i--) /* hidden layers */
    {
        std::cout << "Layer: " << i << '\n';
        Layer* curr = layers[  i  ];
        Layer* next = layers[i + 1];

        VectorXd errors = VectorXd::Zero(curr->layer_size);
      
        for(int j = 0 ; j < curr->layer_size ; j++)
        {
            std::cout << "fiufiu\n";
            double error = 0.0;
            for(int k = 0 ; k < next->layer_size ; k++)
            {
                std::cout << "barabara\n";

                // std::cout << "row: " << next->weights.row(j).size() << '\n';
                std::cout << "delta: " << next->delta.size() << '\n';
                // error += next->weights(j, k) * next->delta[k];
                error += next->weights(k, j) * next->delta[k];
            }
            
            std::cout << "bahbah\n";
            errors[j] = error;
        }
    
    
        for(int j = 0 ; j < curr->layer_size ; j++)
        {
            curr->delta[j] = errors[j] * curr->derivative_outputs[j];
        }
    
    }
}

void NeuralNet::update_weights(float rate)
{
    for(int i = 1 ; i < layers.size() ; i++)
    {
        Layer* curr = layers[  i  ];
        Layer* prev = layers[i - 1];
    
        std::cout << "bah w dupeczke\n";
        curr->weights -= rate * curr->delta * prev->outputs.transpose();
        curr->biases -= rate * curr->delta;
    }

    // for(int i = 1 ; i < layers.size() ; i++)
    // {
    //     VectorXd inputs(layers[i]->outputs.size());
    //     inputs << layers[i]->outputs;

    //     for(int j = 0 ; j < layers[i]->layer_size ; j++)
    //     {
    //         for(int k = 0 ; k < inputs.size() ; k++)
    //         {
    //             //layers[i]->weights.row(k) -= rate * layers[i]->delta * inputs[k];
    //             layers[i]->weights.col(k) -= rate * layers[i]->delta * inputs[k];
    //         }
    //         layers[i]->weights(j, -1) -= rate * layers[i]->delta[j];
    //     }
    // }
}

json NeuralNet::export_to_json()
{
    json obj;

    obj["Network"] = {};

    for(int i = 0 ; i < layers.size() ; i++)
    {
        std::string layerKey = "Layer" + std::to_string(i);
        obj["Network"][layerKey] = {};

        for(int j = 0 ; j < layers[i]->layer_size ; j++)
        {
            std::string neuronKey = "Neuron" + std::to_string(j);
            
            double bias = layers[i]->biases(j);
            std::vector<double> weights_vec(layers[i]->weights.row(j).data(), layers[i]->weights.row(j).data() + layers[i]->weights.row(j).size());
            std::vector<double> outputs_vec;
            if(i == 0)
            {
                outputs_vec = std::vector<double>();
            }
            else
            {
                outputs_vec = std::vector<double>(layers[i]->outputs.data(),
                                                  layers[i]->outputs.data() + layers[i]->outputs.size());
            }
                
            obj["Network"][layerKey][neuronKey] = 
            {
                {"bias", bias},
                {"weights", weights_vec},
                {"outputs", outputs_vec}
            };
        }
    }

    return obj;
}
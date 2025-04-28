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
            this->backpropagate(y, rate);

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

void NeuralNet::backpropagate(VectorXd expected, double rate)
{
    Layer* output_layer = layers.back();

    //VectorXd network_error = (output_layer->outputs - expected) * output_layer->derivative_outputs;

    for(int i = layers.size() - 1 ; i > 0 ; i--)
    {
        std::cout << "Layer: " << i << '\n';
        Layer* curr = layers[  i  ];
        // VectorXd errors(curr->layer_size);
        VectorXd errors = VectorXd::Zero(curr->layer_size);

        if(curr == layers.back()) /*  */
        {
            std::cout << "dudu\n";
            for(int j = 0 ; j < curr->layer_size ; j++)
            {
                std::cout << "errors size:" << errors.size() << '\n';
                std::cout << "output size:" << curr->outputs.size() << '\n';
                std::cout << "target size:" << expected.size() << '\n';                
                errors[j] = curr->outputs[j] - expected[j];
            }
            
            for(int j = 0 ; j < curr->layer_size ; j++)
            {
                std::cout << "brrrr\n";
                std::cout << "errors size:" << errors.size() << '\n';
                std::cout << "delta size:" << curr->delta.size() << '\n';
                std::cout << "derivatives size:" << curr->derivative_outputs.size() << '\n';
                curr->delta[j] = errors[j] * curr->derivative_outputs[j];
            }

            continue;
        }

        Layer* next = layers[i + 1];


        for(int j = 0 ; j < curr->layer_size ; j++)
        {
            std::cout << "fiufiu\n";
            double error = 0.0;
            for(int k = 0 ; k < next->layer_size ; k++)
            {
                std::cout << "barabara\n";
                // std::cout << "barabara" << << '\n';
                // std::cout << "barabara" << <<'\n';
                std::cout << "row: " << next->weights.row(j).size() << '\n';
                std::cout << "delta: " << next->delta.size() << '\n';
                // error += next->weights(j, k) * next->delta[k];
                error += next->weights.row(k)[j] * next->delta[k];
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

// void NeuralNet::backpropagate(VectorXd delta, double rate)
// {
//     VectorXd network_output = this->layers.back()->outputs;

//     std::cout << "brrrr\n";
//     for(int i = layers.size() - 1 ; i > 0 ; i--)
//     {
//         std::cout << i << '\n';
//         delta = layers[i]->backprop(delta, rate);
//     }
    
//     /*
//     VectorXd output =  this->layers.back()->outputs;
//     //VectorXd output(this->layers.back()->layer_size);
//     //output << this->layers.back()->outputs;
    
//     VectorXd delta  = expected - output;
    
//     for(int i = layers.size() - 1 ; i > 0 ; i--)
//     {
//         // Layer* prev = this->layers[i - 1];
//         Layer* prev = i > 0 ? this->layers[i - 1] : nullptr;
//         Layer* curr = this->layers[  i  ];
//         Layer* next = this->layers.size() < i ? this->layers[i + 1] : nullptr;

//         VectorXd errors = VectorXd::Zero(curr->outputs.size());

//         if(i != layers.size() - 1)
//         {

//             for(int j = 0 ; j < curr->outputs.size() ; j++)
//             {
//                 double error = 0.0;
//                 for(int k = 0 ; k < next->outputs.size() ; k++)
//                 {
//                     error += next->weights.row(j).transpose()[k] * next->delta[k];
//                 }

//                 //errors[j] = error;
//                 //errors.push_back(error);
//             }
//         }
//         else
//         {
//             errors = curr->outputs - expected;
//         }

//         for(int j = 0 ; j < curr->outputs.size() ; j++)
//         {
//             //curr->delta[j] = errors[j] * (curr->outputs[j] * (1 - curr->outputs[j]));
//             curr->delta[j] = errors[j] * curr->activation_function(curr->outputs[j]);
//         }
//     }
//     */
// }

/* void NeuralNet::update_weights(float rate)
{
    for(int i = 0 ; i < layers.size() ; i++)
    {
        VectorXd inputs(layers[i]->outputs.size());
        inputs << layers[i]->outputs;

        for(int j = 0 ; j < layers[i]->layer_size ; j++)
        {
            for(int k = 0 ; k < inputs.size() ; k++)
            {
                //layers[i]->weights.row(k) -= rate * layers[i]->delta * inputs[k];
                layers[i]->weights.col(k) -= rate * layers[i]->delta * inputs[k];
            }
            layers[i]->weights(j, -1) -= rate * layers[i]->delta[j];
        }
    }
} */

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
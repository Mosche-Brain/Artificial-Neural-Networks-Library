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
}

NeuralNet::NeuralNet(std::vector<Layer*> topology)
{
    this->layers = topology;
}

NeuralNet::NeuralNet(std::vector<INIT_PARAMS> topology)
{
    this->layers.push_back( new Layer(topology[0].size, 1, topology[0].activation, topology[0].derivative));
    int input_width = topology[0].size;

    for(int i = 1 ; i < topology.size() ; i++)
    {
        input_width = layers[i - 1]->size();
        this->layers.push_back( new Layer(topology[i].size, input_width, topology[i].activation, topology[i].derivative));
    }
}

void NeuralNet::setLossFunction(std::function<double(VectorXd, VectorXd)> func)
{
    this->loss_function = func;
}

void NeuralNet::train(MatrixXd train_x, VectorXd train_y, uint n_iter, float rate)
{
    for(int epoch = 0 ; epoch < n_iter ; epoch++)
    {
        // std::cout << "Epoch " << epoch << " started\n";
        
        std::println("Epoch {}", epoch);

        for(int i = 0 ; i < train_x.rows() ; i++)
        {
            VectorXd x = train_x.row(i).transpose();
            VectorXd y = train_y.row(i).transpose();

            VectorXd predicted = this->forward(x, true);                        
            
            this->backpropagate(y);
            this->update_weights(rate);

            // std::cout << "Row " << i << " processed\n";
        }
    }

    // std::cout << "Training ok\n";
    std::println("Training succes")
; }


VectorXd NeuralNet::forward(MatrixXd input, bool derivatives)
{
    if(input.rows() == 1 && input.cols() > 1)
        input.transposeInPlace();

    layers[0]->outputs = input;

    for(int i = 1 ; i < this->layers.size() ; i++)
    {
        this->layers[i]->forward(layers[i - 1]->outputs, derivatives);
    }
    
    return this->layers.back()->outputs;
}

MatrixXd NeuralNet::predict(MatrixXd input)
{
    MatrixXd output(input.rows(), layers.back()->size());

    for(int i = 0 ; i < input.rows() ; i++)
    {
        output.row(i) = forward(input.row(i));
    }

    return output;
}


void NeuralNet::backpropagate(VectorXd expected)
{
    Layer* output_layer = layers.back();
    
    if(expected.size() != output_layer->outputs.size())
    {
        // std::cout << "Output layer size doesn't match with train target\n";
        std::println("Output layer size doesn't match with train target");
        return;
    }

    VectorXd errors = output_layer->outputs - expected;
    output_layer->delta = output_layer->derivative_outputs.cwiseProduct(errors);
   
    for(int i = layers.size() - 2 ; i > 0 ; i--) /* hidden layers */
    {
        Layer* curr = layers[  i  ];
        Layer* next = layers[i + 1];

        VectorXd errors = VectorXd::Zero(curr->size());
      
        errors = (next->weights.array().colwise() * next->delta.array()).colwise().sum();
        curr->delta = curr->derivative_outputs.cwiseProduct(errors);  
    }
}

void NeuralNet::update_weights(float rate)
{
    for(int i = 1 ; i < layers.size() ; i++)
    {
        Layer* curr = layers[  i  ];
        Layer* prev = layers[i - 1];
    
        curr->weights -= rate * curr->delta * prev->outputs.transpose();
        curr->biases -= rate * curr->delta;
    }
}

void NeuralNet::export_to_json(const char* filename)
{
    json obj;

    obj["Network"] = {};

    for(int i = 0 ; i < layers.size() ; i++)
    {
        std::string layerKey = "Layer" + std::to_string(i);
        obj["Network"][layerKey] = {};

        for(int j = 0 ; j < layers[i]->size() ; j++)
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


    std::ofstream file(filename);
    if(file.is_open())
    {
        file << obj.dump(4);
        file.close();
    }
}

void NeuralNet::load_from_json(const char* filename)
{
    json obj;

    std::ifstream file(filename);
    if(file.is_open())
    {
        obj = json::parse(file);
        file.close();
    }

    std::vector<Layer*> topology;
    
    for (uint i = 0 ; i < obj.contains("Layer" + std::to_string(i)) ; ++i)
    {
        auto layer_data = obj["Layer" + std::to_string(i)];
        int layer_size = layer_data.size(); // Liczba neuronów w warstwie
        int input_size = 0;


        // Utwórz warstwę
        Layer* layer;
        if (i == 0)
        {
            // Warstwa wejściowa (pass-through)
            layer = new Layer(layer_size, input_size, pass, pass_prim, true);
        }
        else
        {
            layer = new Layer(layer_size, input_size, sigmoid, sigmoid_prim);
        }

        // Ustaw wagi i biasy dla każdego neuronu
        for (int j = 0; j < layer_size; ++j) {
            auto neuron_data = layer_data["Neuron" + std::to_string(j)];
            double bias = neuron_data["bias"].get<double>();
            std::vector<double> weights_vec = neuron_data["weights"].get<std::vector<double>>();

            // Ustaw bias
            layer->biases(j) = bias;
            
            // Ustaw wagi
            for (size_t k = 0; k < weights_vec.size(); ++k) {
                layer->weights(j, k) = weights_vec[k];
            }
        }
        
        topology.push_back(layer);
    }
    
    this->layers.clear();
    this->layers = topology;
}
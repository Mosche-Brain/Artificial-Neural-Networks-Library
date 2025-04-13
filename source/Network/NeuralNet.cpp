#include "Network/NeuralNet.hpp"

NeuralNet::NeuralNet(int layers_n, VectorXi layers_size)
{
    if(layers_n != layers_size.size())
        return;

    this->layers.push_back(new Layer(layers_size[0], 1, linear_standard));
    //this->layers[0]->weights = MatrixXd::Ones();

    for(int i = 1 ; i < layers_n ; i++)
    {
        this->layers.push_back(new Layer(layers_size[i], layers_size[i - 1], sigmoid));
    }



    // for(int i = 0 ; i < layers.size() ; i++)
    // {
    //     std::cout << "Layer size: " << layers[i]->layer_size << " Input size: " << layers[i]->input_size << '\n'; 
    // }
    
}

NeuralNet::NeuralNet(std::vector<Layer*> _layers_)
{
    this->layers = _layers_;
}

void NeuralNet::train(MatrixXd train_data, VectorXd expected, uint n_iter, float rate)
{
    for(int epoch = 0 ; epoch < n_iter ; epoch++)
    {
        int error_sum = 0;

        for(int i = 0 ; i < train_data.rows() ; i++)
        {
            RowVectorXd row = train_data.row(i);

            VectorXd outputs = this->forward(row);

            this->backpropagate(expected, rate);
            this->update_weights(rate);
        }
    }

    std::cout << "Training ok\n";
}

VectorXd NeuralNet::forward(VectorXd input)
{
    for(int i = 0 ; i < this->layers.size() ; i++)
    {
        //layers
    }
}

VectorXd NeuralNet::forward(MatrixXd input)
{
    std::cout << "Size: " << input.cols() << '\n';
    std::cout << "cols: " << input.cols() << '\n';
    std::cout << "rows: " << input.rows() << '\n';

    layers[0]->outputs = input.transpose();

    std::cout << "ok\n";
    for(int i = 0 ; i < this->layers.size() - 1 ; i++)
    {
        std::cout << i << " Iteration\n";
        //input = layers[i]->outputs;

        //layers[i + 1]->outputs = layers[i + 1]->forward(layers[i]->outputs);
        this->layers[i + 1]->forward(layers[i]->outputs.transpose());
    }
    
    return this->layers.back()->outputs;
}

VectorXd NeuralNet::forward(int currentIndex, int nextIndex)
{

    Layer* currentLayer = this->layers[currentIndex];
    Layer* nextLayer    = this->layers[nextIndex];

    VectorXd input = currentLayer->outputs;
    VectorXd output = nextLayer->forward(input);
 
    if(nextIndex >= this->layers.size() - 1)
        return output;

    return this->forward(currentIndex + 1, nextIndex + 1);
}

void NeuralNet::backpropagate(VectorXd expected, double rate)
{
    VectorXd output =  this->layers.back()->outputs;
    //VectorXd output(this->layers.back()->layer_size);
    //output << this->layers.back()->outputs;

    VectorXd delta  = expected - output;

    for(int i = layers.size() - 1 ; i > 0 ; i--)
    {
        // Layer* prev = this->layers[i - 1];
        Layer* prev = i > 0 ? this->layers[i - 1] : nullptr;
        Layer* curr = this->layers[  i  ];
        //Layer* next = this->layers[i + 1];
        Layer* next = this->layer_size < i ? this->layers[i + 1] : nullptr;

        //VectorXd errors(curr->outputs.size());
        //std::vector<double> errors(curr->outputs.size());
        VectorXd errors = VectorXd::Zero(curr->outputs.size());

        if(i != layers.size() - 1)
        {

            for(int j = 0 ; j < curr->outputs.size() ; j++)
            {
                double error = 0.0;
                for(int k = 0 ; k < next->outputs.size() ; k++)
                {
                    error += next->weights.row(j).transpose()[k] * next->delta[k];
                }

                //errors[j] = error;
                //errors.push_back(error);
            }
        }
        else
        {
            // for(int j = 0 ; j < curr->outputs.size() ; j++)
            // {
            //     //errors.push_back(curr->outputs[j] - expected[j]);
            // }
            errors = curr->outputs - expected;
        }

        for(int j = 0 ; j < curr->outputs.size() ; j++)
        {
            //curr->delta[j] = errors[j] * (curr->outputs[j] * (1 - curr->outputs[j]));
            curr->delta[j] = errors[j] * curr->activation_function(curr->outputs[j]);
        }

        /*
        std::cout << "okej1\n";
        curr->biases += delta * -rate;
        std::cout << "okej2\n";
        //prev->weights += -rate * (delta * prev->outputs.transpose());

        
        std::cout << "Weight Cols " << prev->weights.cols() << '\n';
        std::cout << "Weight Rows " << prev->weights.rows() << '\n';
        std::cout << "Output Cols " << prev->outputs.cols() << '\n';
        std::cout << "Output Rows " << prev->outputs.rows() << '\n';
        std::cout << "Delta Cols " << delta.cols() << '\n';
        std::cout << "Delta Rows " << delta.rows() << '\n';
        //std::cout << "Outputs " << prev->outputs.size();
        
        std::cout << "okej69\n";
        prev->weights += -rate * (delta * prev->outputs);
        std::cout << "okej3\n";
        //std::cout << -rate * (delta * prev->outputs.transpose()) << '\n';
        std::cout << "okej4\n";

        //delta = (curr->weights.transpose() * delta).array() * (prev->outputs.array() * (1 - prev->outputs.array()));
        // MatrixXd mat = MatrixXd::Ones(prev->outputs.rows(), prev->outputs.cols());
        // MatrixXd sigmatrix = prev->outputs * (mat - prev->outputs);


        std::cout << "okej\n";
        VectorXd vec = VectorXd::Ones(prev->outputs.size());
        VectorXd sigmatrix = prev->outputs * (vec - prev->outputs);

        delta = delta.array() * sigmatrix.array();
        */
    }
}

void NeuralNet::update_weights(float rate)
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
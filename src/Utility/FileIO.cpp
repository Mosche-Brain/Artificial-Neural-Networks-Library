#include "FileIO.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace YANN::Utils
{

    void FileIO::saveMatrix(YANN::matrix_t mat, const char* filename)
    {
        std::ofstream file(filename);
        if(file.is_open())
        {
            for(size_t i = 0; i < YANN::math_api::matrixRows(mat); ++i)
            {
                for(size_t j = 0; j < YANN::math_api::matrixCols(mat); ++j)
                {
                    file << std::to_string(mat(i, j)) << " ";
                }
                file << "\n";
                // file << std::to_string(x(i)) << " " << std::to_string(y_true(i)) << " " << std::to_string(y_pretrain(i)) << " " << std::to_string(y_pred(i)) << "\n";
            }
            file.close();
        }
        else
        {
            std::cerr << "Unable to open file for writing\n";
        }
    }
    
    void FileIO::readMatrix(YANN::matrix_t& mat, const char* filename)
    {
        std::ifstream file(filename);
        if(file.is_open())
        {
            std::vector<std::vector<numeric_t>> data;
            std::string line;
            size_t cols = 0;

            while(std::getline(file, line))
            {
                std::istringstream iss(line);
                std::vector<numeric_t> row;
                numeric_t value;
                while(iss >> value)
                {
                    row.push_back(value);
                }
                if (!row.empty())
                {
                    if (cols == 0)
                        cols = row.size();
                    else if (row.size() != cols)
                    {
                        std::cerr << "Inconsistent number of columns in file\n";
                        file.close();
                        return;
                    }
                    data.push_back(row);
                }
            }

            size_t rows = data.size();
            mat.resize(rows, cols);
            for(size_t i = 0; i < rows; ++i)
            {
                if (data[i].size() != cols)
                {
                    std::cerr << "Row " << i << " has " << data[i].size() << " columns, expected " << cols << ". Aborting matrix load.\n";
                    mat.resize(0, 0);
                    file.close();
                    return;
                }
                for(size_t j = 0; j < cols; ++j)
                {
                    mat(i, j) = data[i][j];
                }
            }
            if (rows == 0) {
                std::cerr << "File is empty or contains no valid data\n";
            }
            file.close();
        }
        else
        {
            std::cerr << "Unable to open file for reading: " << filename << "\n";
        }
    }

    void FileIO::saveSequentialModel(const Models::Sequential& model, const char* filename)
    {
        // model format: number of layers, for each layer: layer type, layer size, weights, biases
        std::ofstream file(filename, std::ios::binary);
        if(file.is_open())
        {
            json j;
            j["layers"] = json::array();

            for(size_t i = 0 ; i < model.getLayersCount() ; i++)
            {
                json layer_json;
                
                matrix_t weights = model.getWeights(i);

                std::vector<std::vector<numeric_t>> weights_vec(weights.rows());
                
                for(int i = 0; i < weights.rows(); ++i)
                    weights_vec[i] = std::vector<numeric_t>(weights.row(i).data(), weights.row(i).data() + weights.cols());
                layer_json["weights"] = weights_vec;
                
                matrix_t biases = model.getBiases(i);
                std::vector<numeric_t> biases_vec(biases.data(), biases.data() + biases.size());
                layer_json["biases"] = biases_vec;
                
                layer_json["activation"] = model.getActivation(i).name;
                j["layers"].push_back(layer_json);
            }

            // convert to bson and write to file
            std::vector<uint8_t> bson_data = json::to_bson(j);
            file.write(reinterpret_cast<const char*>(bson_data.data()), bson_data.size());

            file.close();
        }
        
    }

    void FileIO::loadSequentialModel(Models::Sequential& model, const char* filename)
    {
        model.clear();
        std::ifstream file(filename, std::ios::binary);
        if(file.is_open())
        {
            std::vector<uint8_t> bson_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            json j = json::from_bson(bson_data);
            for(const auto& layer_json : j["layers"])
            {
                matrix_t weights = math_api::createMatrix(layer_json["weights"].size(), layer_json["weights"][0].size());
                for(int i = 0; i < weights.rows(); ++i)
                    for(int j = 0; j < weights.cols(); ++j)
                        weights(i, j) = layer_json["weights"][i][j].get<numeric_t>();

                matrix_t biases = math_api::createMatrix(layer_json["biases"].size(), 1);
                for(int i = 0; i < biases.rows(); ++i)
                    biases(i, 0) = layer_json["biases"][i].get<numeric_t>();

                Utils::activation_t activation;
                activation.name = layer_json["activation"].get<std::string>().c_str();

                // create layer and set parameters
                auto layer = Models::Layers::Dense::createUnique(weights.rows(), activation.name);
                layer->Weights() = weights;
                layer->Biases() = biases;
                model.addLayer(std::move(layer));
            }
            file.close();
        }
    }
}
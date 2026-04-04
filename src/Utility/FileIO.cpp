#include "FileIO.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "YANN/Models/Sequential.hpp"

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

            for(const auto& layer : model.getTopology())
            {
                json layer_json;
                // layer_json["type"] = layer->getType();
                // layer_json["size"] = layer->getSize();
                
                // Convert Eigen matrices to vectors for JSON serialization
                auto weights = layer->Weights();
                std::vector<std::vector<numeric_t>> weights_vec(weights.rows());
                for(int i = 0; i < weights.rows(); ++i)
                    weights_vec[i] = std::vector<numeric_t>(weights.row(i).data(), weights.row(i).data() + weights.cols());
                layer_json["weights"] = weights_vec;
                
                auto biases = layer->Biases();
                std::vector<numeric_t> biases_vec(biases.data(), biases.data() + biases.size());
                layer_json["biases"] = biases_vec;
                
                layer_json["activation"] = layer->activation.name;
                j["layers"].push_back(layer_json);
            }


        
            file.close();
        }
        
    }

    void FileIO::loadSequentialModel(Models::Sequential& model, const char* filename)
    {
        /*
        std::ifstream file(filename, std::ios::binary);
        if(file.is_open())
        {
            model.deserialize(file);
            file.close();
        }
        */
    }
}
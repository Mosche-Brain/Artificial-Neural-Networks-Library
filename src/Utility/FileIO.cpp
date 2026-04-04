#include "FileIO.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

#include "YANN/Models/Sequential.hpp"

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
        /*
        std::ofstream file(filename, std::ios::binary);
        if(file.is_open())
        {            model.serialize(file);
            file.close();
        }
        */
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
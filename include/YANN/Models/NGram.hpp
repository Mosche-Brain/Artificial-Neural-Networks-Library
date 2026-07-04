#pragma once
#include <eigen3/Eigen/Dense>

#include <unordered_map>
#include <unordered_set>

#include <vector>
#include <string>

#include <fstream>
#include <iostream>

#define int_t int32_t

namespace yann::models
{
    class NGram
    {
    protected:
        std::unordered_map<std::string, int_t> token_to_id;  // Maps string → unique ID
        std::unordered_map<int_t, std::string> id_to_token;  // Maps unique ID → string
        std::unordered_set<int_t> tokens;  // Just store the IDs
        std::unordered_map<int_t, std::unordered_map<int_t, size_t>> probability_map;

        std::string raw_data;
        void tokenize();
    public:
        NGram();
        ~NGram();

        void fit(const std::string& text);

        int_t predict(const int_t& token, size_t top_k = 5);
        std::string predict(const std::string& token, size_t top_k = 5);

        std::unordered_map<int_t, size_t> getNextTokenProbabilities(const int_t& token);
        std::unordered_map<std::string, size_t> getNextTokenProbabilities(const std::string& token);



        void saveModel(const char* path);
        void loadModel(const char* path);
    };

}


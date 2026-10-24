/**
 * @file NGram.hpp
 * @brief Statistical N-gram language model for text tokenization and next-token prediction.
 */

#pragma once

#include <eigen3/Eigen/Dense>

#include <unordered_map>
#include <unordered_set>

#include <vector>
#include <string>
#include <cstdint>

#include <fstream>
#include <iostream>

#define int_t int32_t

namespace yann::models
{
    /**
     * @class NGram
     * @brief N-gram probabilistic language model tracking transition frequencies between tokens.
     */
    class NGram
    {
    protected:
        std::unordered_map<std::string, int_t> token_to_id;  /**< Map from string token to token ID. */
        std::unordered_map<int_t, std::string> id_to_token;  /**< Map from token ID back to string token. */
        std::unordered_set<int_t> tokens;                    /**< Set of all unique token IDs encountered. */
        std::unordered_map<int_t, std::unordered_map<int_t, size_t>> probability_map; /**< Co-occurrence count map between tokens. */

        std::string raw_data;                                /**< Raw text corpus. */

        /**
         * @brief Tokenize raw text into individual vocabulary tokens and build token mappings.
         */
        void tokenize();

    public:
        /**
         * @brief Default constructor for NGram model.
         */
        NGram();

        /**
         * @brief Destructor.
         */
        ~NGram();

        /**
         * @brief Fit the language model on a raw text corpus.
         * @param text String containing the training text.
         */
        void fit(const std::string& text);

        /**
         * @brief Predict the next token ID given a current token ID.
         * @param token ID of the current token.
         * @param top_k Consider top-K candidates during prediction (default: 5).
         * @return Most probable next token ID.
         */
        int_t predict(const int_t& token, size_t top_k = 5);

        /**
         * @brief Predict the next string token given a current string token.
         * @param token String representation of the current token.
         * @param top_k Consider top-K candidates during prediction (default: 5).
         * @return Predicted next token as string.
         */
        std::string predict(const std::string& token, size_t top_k = 5);

        /**
         * @brief Get counts/frequencies of tokens following a given token ID.
         * @param token Current token ID.
         * @return Map of successor token IDs to their occurrence frequencies.
         */
        std::unordered_map<int_t, size_t> getNextTokenProbabilities(const int_t& token);

        /**
         * @brief Get counts/frequencies of tokens following a given string token.
         * @param token Current string token.
         * @return Map of successor string tokens to their occurrence frequencies.
         */
        std::unordered_map<std::string, size_t> getNextTokenProbabilities(const std::string& token);

        /**
         * @brief Serialize model vocabulary and transition counts to disk.
         * @param path Target file path.
         */
        void saveModel(const char* path);

        /**
         * @brief Load model vocabulary and transition counts from disk.
         * @param path Source file path.
         */
        void loadModel(const char* path);
    };

} // yann::models

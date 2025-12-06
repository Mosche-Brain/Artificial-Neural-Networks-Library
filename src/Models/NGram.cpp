#include "NGram.hpp"

#include <cctype>
#include <algorithm>
#include <random>

namespace ANN::Models
{

    NGram::NGram(/* args */)
    {
    }

    NGram::~NGram()
    {
    }

    void NGram::tokenize()
    {
        std::vector<std::string> sentences;
        std::string sentence;
        
        // Split by sentences
        for(auto& c : raw_data)
        {
            if(c == '.')
            {
                if(!sentence.empty()) sentences.push_back(sentence);
                sentence = "";
            }
            else
                sentence += c;
        }
        if(!sentence.empty()) sentences.push_back(sentence);

        std::cout << sentences.size() << " sentences\n";

        // Tokenize and build maps in single pass
        for (const auto& sent : sentences)
        {
            std::vector<std::string> tokenized_sentence;
            std::string token;
            
            for(auto& c : sent)
            {
                if((std::isblank(c) || std::ispunct(c)) && !token.empty())
                {
                    tokenized_sentence.push_back(token);
                    token = "";
                }
                else
                {
                    token += c;
                }
            }
            if(!token.empty()) tokenized_sentence.push_back(token);

            // Add tokens to set and build probability map in one loop
            int_t next_id = static_cast<int_t>(token_to_id.size());
            for (size_t j = 0; j < tokenized_sentence.size(); j++)
            {
                const auto& tok_str = tokenized_sentence[j];

                // assign/get id for current token
                auto it = token_to_id.find(tok_str);
                int_t token_id;
                if (it == token_to_id.end())
                {
                    token_id = next_id++;
                    token_to_id.emplace(tok_str, token_id);
                    id_to_token.emplace(token_id, tok_str);
                }
                else
                {
                    token_id = it->second;
                }

                tokens.insert(token_id);
                
                if (j < tokenized_sentence.size() - 1)
                {
                    const auto& next_str = tokenized_sentence[j + 1];
                    auto it2 = token_to_id.find(next_str);
                    int_t next_token_id;
                    if (it2 == token_to_id.end())
                    {
                        next_token_id = next_id++;
                        token_to_id.emplace(next_str, next_token_id);
                        id_to_token.emplace(next_token_id, next_str);
                    }
                    else
                    {
                        next_token_id = it2->second;
                    }

                    probability_map[token_id][next_token_id]++;
                }
            }
        }

        std::cout << "Total unique tokens: " << tokens.size() << "\n";
    }
    
    void NGram::fit(const std::string& text)
    {
        raw_data = text;

        this->tokenize();
    }

    int_t NGram::predict(const int_t& token, size_t top_k)
    {
        std::unordered_map<int_t, size_t> next_token_probs = getNextTokenProbabilities(token);
        if (next_token_probs.empty()) return -1; // or some invalid token ID
        // Find the top k tokens
        std::vector<std::pair<int_t, size_t>> sorted_tokens(next_token_probs.begin(), next_token_probs.end());
        std::sort(sorted_tokens.begin(), sorted_tokens.end(), [](const auto& a, const auto& b) {
            return a.second > b.second; });

        if (top_k > sorted_tokens.size()) top_k = sorted_tokens.size();

        // return token adaccording to probability distribu
        
    
    }

    std::unordered_map<int_t, size_t> NGram::getNextTokenProbabilities(const int_t& token)
    {
        auto it = probability_map.find(token);
        if (it != probability_map.end())
        {
            return it->second;
        }
        return {};
    }

    std::unordered_map<std::string, size_t> NGram::getNextTokenProbabilities(const std::string& token)
    {
        auto it = token_to_id.find(token);
        if (it != token_to_id.end())
        {
            int_t token_id = it->second;
            auto probs = getNextTokenProbabilities(token_id);
            std::unordered_map<std::string, size_t> result;
            for (const auto& [next_id, count] : probs)
            {
                result[id_to_token[next_id]] = count;
            }
            return result;
        }
        return {};
    }    

    std::string NGram::predict(const std::string& token, size_t top_k)
    {

        return id_to_token[predict(token_to_id[token], top_k)]; // Placeholder
    }

    void NGram::saveModel(const char* path)
    {

    }

    void NGram::loadModel(const char* path)
    {

    }
}
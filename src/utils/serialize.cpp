/*
 * @author: jaro
 * @name:   serialize
 * @file:   src/utils/serialize.cpp
 * @date:   29 September 2026 09:32:24
 */

#include <ranges>
#include <print>
#include <format>

#include "yann/models/Sequential.hpp"

#include "yann/utils/serialize.hpp"

namespace yann::utils
{
    void model_serialize_to_safetensors(const models::Sequential& model, const char* filename)
    {
        std::unordered_map<std::string, cum::Tensor> model_tensors;

        for (auto [index, layer] : model.getTopology() | std::views::enumerate)
        {
            if (layer->layerType() == models::layers::LayerType::Input || layer->layerType() == models::layers::LayerType::Softmax)
                continue;

            std::string weights_tag = std::to_string(index) + '_' + 'w';
            std::string biases_tag = std::to_string(index) + '_' + 'b';

            cum::Tensor& weights = layer->weights();
            cum::Tensor& biases = layer->biases();

            cum::Shape weights_indices = cum::Shape(weights.rank(), 0);
            for (int i = 0; i < weights.rank(); i++)
            {
                for ( ; weights_indices[i]  < weights.shape()[i]; weights_indices[i]++)
                {
                    std::print("{} ", weights.at<float>(weights_indices));
                }
            }
            std::println();
            cum::Shape biases_indices = cum::Shape(biases.rank(), 0);
            for (int i = 0; i < weights.rank(); i++)
            {
                for ( ; biases_indices[i]  < biases.shape()[i]; biases_indices[i]++)
                {
                    std::print("{} ", biases.at<float>(biases_indices));
                }
            }
            std::println();
            std::println("--------------------------------");

            model_tensors.emplace(weights_tag, weights);
            model_tensors.emplace(biases_tag, biases);
        }

        save_safetensors(model_tensors, filename);

    }

    // models::Sequential model_deserialize_from_safetensors(const char* filename)
    void model_deserialize_from_safetensors(models::Sequential& model, const char* filename)
    {
        std::unordered_map<std::string, cum::Tensor> model_tensors = load_safetensors(filename);

        for (auto [index, layer] : model.getTopology() | std::views::enumerate)
        {
            if (layer->layerType() == models::layers::LayerType::Input)
                continue;

            std::string weights_tag = std::to_string(index) + '_' + 'w';
            std::string biases_tag = std::to_string(index) + '_' + 'b';

            cum::Tensor weights = model_tensors[weights_tag];
            cum::Tensor biases = model_tensors[biases_tag];

            cum::Shape weights_indices = cum::Shape(weights.rank(), 0);
            for (int i = 0; i < weights.rank(); i++)
            {
                for ( ; weights_indices[i]  < weights.shape()[i]; weights_indices[i]++)
                {
                    std::print("{} ", weights.at<float>(weights_indices));
                }
            }
            std::println();
            cum::Shape biases_indices = cum::Shape(biases.rank(), 0);
            for (int i = 0; i < weights.rank(); i++)
            {
                for ( ; biases_indices[i]  < biases.shape()[i]; biases_indices[i]++)
                {
                    std::print("{} ", biases.at<float>(biases_indices));
                }
            }
            std::println();
            std::println("--------------------------------");

            layer->weights() = weights;
            layer->biases() = biases;
        }
    }

}
//
// Created by jaro on 7/2/26.
//

#pragma once

#include <cum/memory.hpp>

#include <vector>
#include <memory>

namespace cum::neural_primitives
{
    typedef struct memory memory;

    class tensor_descriptor
    {
    public:
        using dimensions = std::vector<int64_t>;
        enum class data_type { FP64, FP32, FP16, BF16, S8, U8 };
        enum class layout { ANY, X, NC, OI, NCHW, NHWC, STRIDED };

        tensor_descriptor(dimensions dims, data_type dtype, layout format=layout::ANY);
        ~tensor_descriptor();
    private:
        struct Impl;
        Impl* impl;
    };

    // class Activation
    // {
        // static void relu();
    // };

}
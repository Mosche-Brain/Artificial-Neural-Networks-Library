//
// Created by jaro on 7/19/26.
//

#ifndef YANN_TENSOR_DESCRIPTOR_HPP
#define YANN_TENSOR_DESCRIPTOR_HPP
#include <cstdint>
#include <vector>

namespace cum::neural_primitives
{

    class tensor_descriptor
    {
    public:

        enum class dtype { FP64, FP32, FP16, BF16, S8, U8 };
        enum class layout { ANY, X, NC, OI, NCHW, NHWC, STRIDED }; // MCHW

        tensor_descriptor(Shape dims, dtype dtype, layout format=layout::ANY);
        ~tensor_descriptor();
    private:
        struct Impl;
        Impl* impl;
    };
} // cum

#endif //YANN_TENSOR_DESCRIPTOR_HPP

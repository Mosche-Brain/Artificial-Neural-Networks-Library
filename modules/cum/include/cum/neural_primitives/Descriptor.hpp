//
// Created by jaro on 7/19/26.
//

#ifndef YANN_TENSOR_DESCRIPTOR_HPP
#define YANN_TENSOR_DESCRIPTOR_HPP
#include <cstdint>
#include <vector>
#include <memory>

#include "cum/Core.hpp"

#include "cum/neural_primitives/opaque_types.hpp"

namespace cum::neural_primitives
{

    class Descriptor
    {
    public:
        Descriptor(Shape dims, datatype dtype, layout format=layout::ANY);
        ~Descriptor();
    private:
        friend class Memory;

        std::unique_ptr<handles::__desc__> handle;
    };
} // cum

#endif //YANN_TENSOR_DESCRIPTOR_HPP

//
// Created by jaro on 7/19/26.
//

#ifndef YANN_TENSOR_MEMORY_HPP
#define YANN_TENSOR_MEMORY_HPP

#include "tensor_descriptor.hpp"
#include "opaque_types.hpp"

namespace cum::neural_primitives
{
    class tensor_memory
    {
    public:
        tensor_memory(const tensor_descriptor& desc, Engine& engine);
        tensor_memory(const tensor_descriptor& desc, Engine& engine, void* data);

        void* data() const;
        void  set_data(void* data);

        const tensor_descriptor& descriptor() const;
    private:
        struct Impl;
        Impl* impl;
    };

} // cum

#endif //YANN_TENSOR_MEMORY_HPP

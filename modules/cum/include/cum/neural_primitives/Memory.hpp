//
// Created by jaro on 7/19/26.
//

#ifndef YANN_TENSOR_MEMORY_HPP
#define YANN_TENSOR_MEMORY_HPP

#include <memory>

#include "cum/neural_primitives/Descriptor.hpp"
#include "cum/neural_primitives/Engine.hpp"
#include "cum/neural_primitives/opaque_types.hpp"

namespace cum::neural_primitives
{
    class Memory
    {
    public:
        Memory(const Descriptor& desc);
        Memory(const Descriptor& desc, void* data);
        Memory(const Descriptor& desc, Engine& engine);
        Memory(const Descriptor& desc, Engine& engine, void* data);

        void* data();
        void  set_data(void* data);

        const Descriptor& descriptor() const;
    private:

        std::unique_ptr<handles::__memory__> handle;
    };

} // cum

#endif //YANN_TENSOR_MEMORY_HPP

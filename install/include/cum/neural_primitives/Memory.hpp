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

        Memory(const Descriptor& desc, const Memory& source);

        Memory(handles::__memory__& handle);

        ~Memory();

        // Make view
        // static Memory make_view(const Memory& memor);

        void* data();
        const void* data() const;

        void  set_data(void* data);

        std::shared_ptr<handles::__memory__> owner() const;

        const handles::__memory__& handle() const;
        handles::__memory__& handle();
        // const Descriptor& descriptor() const;
    private:
        friend class Tensor;

        std::shared_ptr<handles::__memory__> handle_;
        std::shared_ptr<handles::__memory__> owner_; // nullptr by default, point only when Memory object is a view over existing memory
    };

} // cum

#endif //YANN_TENSOR_MEMORY_HPP

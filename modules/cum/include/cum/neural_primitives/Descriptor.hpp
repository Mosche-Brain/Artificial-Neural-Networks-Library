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
        Descriptor(Shape dims, datatype dtype, layout format);

        dim_t size() const;
        dim_t ndims() const;
        Shape shape() const;
        datatype type() const;
        layout format() const;

        const handles::__desc__& handle() const;

        ~Descriptor();
    private:
        friend class Memory;
        friend class Tensor;

        layout fmt;
        std::unique_ptr<handles::__desc__> handle_;
    };
} // cum

#endif //YANN_TENSOR_DESCRIPTOR_HPP

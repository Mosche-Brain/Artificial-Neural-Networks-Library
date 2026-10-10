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
        Descriptor(Descriptor& other);
        Descriptor(const handles::__desc__& handle);
        Descriptor(const handles::__desc__& handle, layout format);
        Descriptor(handles::__desc__& handle);

        Descriptor(Descriptor&&) noexcept;
        Descriptor& operator=(Descriptor&&) noexcept;

        dim_t size() const;
        dim_t ndims() const;
        Shape shape() const;
        datatype type() const;
        layout format() const;
        dim_t stride() const;
        dim_t offset() const;

        Descriptor reshape(const Shape& shape);
        Descriptor& reshape_in_place(const Shape& shape);


        const handles::__desc__& handle() const;
        handles::__desc__& handle();

        Descriptor& operator = (Descriptor& other);
        ~Descriptor();
    private:
        friend class Memory;
        friend class Tensor;

        std::unique_ptr<handles::__desc__> handle_;
        layout fmt;
    };
} // cum

#endif //YANN_TENSOR_DESCRIPTOR_HPP

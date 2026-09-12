//
// Created by jaro on 6/28/26.

#pragma once

#include <initializer_list>
#include <memory>

#include "cum/Core.hpp"
#include "cum/neural_primitives/Descriptor.hpp"
#include "cum/neural_primitives/Memory.hpp"
#include "cum/neural_primitives/opaque_types.hpp"

namespace cum
{
    class Tensor
    {
    public:
        Tensor(Shape shape, datatype dtype = datatype::FP32, layout layout = layout::ANY); // FP32, ANY
		~Tensor();

        Tensor(const Tensor& tensor);
        Tensor(Tensor&& tensor) noexcept;

        template<datatype T, layout layout>
        static Tensor create_tensor(Shape shape)
        {
            return Tensor(shape, T, layout);
        }

        static Tensor take_memory(Shape shape, void* data, datatype dtype = datatype::FP32, layout layout = layout::ANY);

        static Tensor make_cube(dim_t width, dim_t height, dim_t deepth, datatype dtype = default_type, layout layout = layout::OI);
        static Tensor make_matrix(dim_t rows, dim_t cols, datatype dtype = default_type, layout layout = layout::OI);
        static Tensor make_vector(dim_t lenght, datatype dtype = default_type, layout layout = layout::x);
        static Tensor make_scalar(datatype dtype = default_type, layout layout = layout::x);

        dim_t size() const;
        dim_t dims() const;
        dim_t lenght() const;
        Shape shape() const;
        layout format() const;
        datatype type() const;

        std::unique_ptr<neural_primitives::Descriptor>& descriptor();
        std::unique_ptr<neural_primitives::Memory>& memory();

        const std::unique_ptr<neural_primitives::Descriptor>& descriptor() const;
        const std::unique_ptr<neural_primitives::Memory>& memory() const;
		 
        cumeric_t& at(const Shape indices);
        
        const cumeric_t* data() const;
        cumeric_t* data();

        cumeric_t sum();

        Tensor& fill(cumeric_t scalar);

        Tensor multiply(const Tensor& tensor);

        Tensor friend operator + (const Tensor& A, const Tensor& B);
        Tensor friend operator - (const Tensor& A, const Tensor& B);
        Tensor friend operator * (const Tensor& A, const Tensor& B);
        Tensor friend operator / (const Tensor& A, const Tensor& B);

        Tensor friend operator + (const Tensor& tensor, cumeric_t scalar);
        Tensor friend operator + (cumeric_t scalar, const Tensor& tensor);
        Tensor friend operator - (const Tensor& tensor, cumeric_t scalar);
        Tensor friend operator - (cumeric_t scalar, const Tensor& tensor);
        Tensor friend operator * (const Tensor& tensor, cumeric_t scalar);
        Tensor friend operator * (cumeric_t scalar, const Tensor& tensor);
        Tensor friend operator / (const Tensor& tensor, cumeric_t scalar);
        Tensor friend operator / (cumeric_t scalar, const Tensor& tensor);

        bool friend operator == (Tensor& A, Tensor& B);
        bool friend operator != (Tensor& A, Tensor& B);
    private:
        std::unique_ptr<neural_primitives::Descriptor> __desc__;
        std::unique_ptr<neural_primitives::Memory> __data__;
    };
} // cum

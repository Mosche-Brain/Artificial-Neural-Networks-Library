//
// Created by jaro on 6/28/26.

#pragma once

#include <memory>

#include "cum/Core.hpp"
#include "cum/neural_primitives/tensor_descriptor.hpp"
#include "cum/neural_primitives/opaque_types.hpp"

namespace cum
{

    class Tensor
    {
    public:
        Tensor(Shape shape, neural_primitives::tensor_descriptor::dtype dtype=1, neural_primitives::tensor_descriptor::layout=0); // FP32, ANY
		~Tensor();

		 
        cumeric_t& at(const Shape indices);
        
        const cumeric_t* data() const;
        cumeric_t* data();

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
        std::unique_ptr<neural_primitives::tensor_descriptor> __desc___;
        std::unique_ptr<neural_primitives::Memory> __data__;
    };
} // cum

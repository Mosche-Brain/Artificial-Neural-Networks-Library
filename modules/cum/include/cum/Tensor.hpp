//
// Created by jaro on 6/28/26.

#pragma once

#include <initializer_list>
#include <memory>
#include <stdexcept>

#include "Matrix.hpp"
#include "cum/Core.hpp"
#include "cum/neural_primitives/Descriptor.hpp"
#include "cum/neural_primitives/Memory.hpp"
#include "cum/neural_primitives/opaque_types.hpp"

namespace cum
{
    class Tensor
    {
    public:
        /* Constructors */
        Tensor(const Shape& shape, datatype dtype = datatype::FP32, layout layout = layout::ANY); // FP32, ANY
		~Tensor();

        Tensor() = default;
        Tensor(const Tensor& tensor);
        Tensor(Tensor&& tensor) noexcept;

        /* fabriques */

        template<datatype T, layout layout>
        static Tensor create_tensor(Shape shape)
        {
            return Tensor(shape, T, layout);
        }

        static Tensor take_memory(Shape shape, void* data, datatype dtype = datatype::FP32, layout layout = layout::ANY);

        static Tensor make_cube(dim_t width, dim_t height, dim_t deepth, datatype dtype = default_type, layout layout = layout::OI);
        static Tensor make_matrix(dim_t rows, dim_t cols, datatype dtype = default_type, layout layout = layout::OI);
        static Tensor make_vector(dim_t lenght, datatype dtype = default_type, layout layout = layout::X);
        static Tensor make_scalar(datatype dtype = default_type, layout layout = layout::X);

        static Tensor Random(Shape shape, cumeric_t min = -1_c, cumeric_t max = 1_c, datatype dtype = datatype::FP32, layout layout = layout::ANY);
        static Tensor Zeros(Shape shape, datatype dtype = datatype::FP32, layout layout = layout::ANY);
        static Tensor Ones(Shape shape, datatype dtype = datatype::FP32, layout layout = layout::ANY);
        static Tensor Linspace(cumeric_t start, cumeric_t end, dim_t num); // vector

        /* getters */

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
		 
        /* accesors */
        template<typename T>
        const T& at(const Shape& indices) const
        {
            if (type() != datatype_of<T>())
                throw std::runtime_error("datatype mismatch");

            const dim_t idx = compute_index(indices);
            return static_cast<const T*>(__data__)[idx];
        }

        cumeric_t at(const Shape& indices);
        
        const cumeric_t* data() const;
        cumeric_t* data();

        Tensor slice(const Shape& indices);

        Tensor& transpose();

        /* reductions */

        cumeric_t sum();

        Tensor colwise_sum();
        Tensor rowwise_sum();
        Tensor channelwise_sum();

        /* elementwise */

        Tensor& fill(cumeric_t scalar);

        Tensor multiply(const Tensor& tensor);
        Tensor cwiseProduct(const Tensor& tensor);

        Tensor sqrt();

        Tensor& scale(cumeric_t scalar);

        /* operator overloads */

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

        Tensor& operator += (const Tensor& other);
        Tensor& operator -= (const Tensor& other);
        Tensor& operator *= (const Tensor& other);
        Tensor& operator /= (const Tensor& other);

        bool friend operator == (Tensor& A, Tensor& B);
        bool friend operator != (Tensor& A, Tensor& B);

        Tensor& operator = (const Tensor& other);
        Tensor& operator = (Tensor&& other) noexcept;
    private:
        dim_t compute_index(const Shape& indices) const;
        void* compute_address(const Shape& indices);

        std::unique_ptr<neural_primitives::Descriptor> __desc__;
        std::unique_ptr<neural_primitives::Memory> __memr__;
        void* __data__;
    };
} // cum

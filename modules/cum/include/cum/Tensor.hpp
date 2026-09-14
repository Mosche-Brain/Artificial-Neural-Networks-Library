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
    enum class Axis
    {
        Batches,
        Channels,
        Depth,
        Height,
        Width,
        Rows = Height,
        Cols = Width,
    };

    class Tensor
    {
    public:
        /* Constructors */
        Tensor(const Shape& shape, datatype dtype = default_type, layout layout = layout::ANY); // FP32, ANY

        Tensor(datatype dtype = default_type, layout layout = layout::ANY); // scalar constructor
        Tensor(dim_t lenght, datatype dtype = default_type, layout layout = layout::X); // vector constructor
        Tensor(dim_t rows, dim_t cols, datatype dtype = default_type, layout layout = layout::NC); // Matrix Constructor
        Tensor(dim_t axis0, dim_t axis1, dim_t axis2, datatype dtype = default_type, layout layout = layout::NCHW); // 3-rd rank tensor constructor
        Tensor(dim_t axis0, dim_t axis1, dim_t axis2, dim_t axis3, datatype dtype = default_type, layout layout = layout::NCHW); // 4-th rank tensor constructor
        Tensor(dim_t axis0, dim_t axis1, dim_t axis2, dim_t axis3, dim_t axis4, datatype dtype = default_type, layout layout = layout::NCDHW); // 5-th rank tensor constructor

        Tensor() = default;
        Tensor(const Tensor& tensor);
        Tensor(Tensor&& tensor) noexcept;
		~Tensor();

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
        static Tensor Zeros(const Shape& shape, datatype dtype = datatype::FP32, layout layout = layout::ANY);
        static Tensor Ones(const Shape& shape, datatype dtype = datatype::FP32, layout layout = layout::ANY);
        static Tensor Linspace(cumeric_t start, cumeric_t end, dim_t num); // vector

        /* Memory */

        Tensor& prefetch();

        Tensor& to_host();

        Tensor& to_device();

        /* getters */

        dim_t size() const;
        dim_t dims() const;
        dim_t lenght() const;
        Shape shape() const;
        layout format() const;
        datatype type() const;

        bool has(Axis axis) const;
        dim_t extent(Axis axis) const;

        dim_t batches() const;
        dim_t channels() const;
        dim_t depth() const;
        dim_t height() const;
        dim_t width() const;
        dim_t rows() const;    // alias height()
        dim_t cols() const;    // alias width()

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
        cumeric_t at(const Shape& indices) const;

        const cumeric_t* data() const; // I should add templated data getter
        cumeric_t* data();


        /* Reshaping */
        Tensor slice(const Shape& indices);

        Tensor transpose();
        Tensor& transpose_in_place();

        /* reductions */

        cumeric_t sum();
        cumeric_t mean();
        cumeric_t amean();

        cumeric_t squaredNorm() {}
        cumeric_t squared_norm() {}

        Tensor colwise_sum();
        Tensor rowwise_sum();
        Tensor channelwise_sum();

        /* elementwise */

        Tensor& fill(cumeric_t scalar);

        Tensor multiply(const Tensor& tensor);
        Tensor cwiseProduct(const Tensor& tensor);

        Tensor sqrt();
        Tensor& sqrt_in_place();

        Tensor square();
        Tensor& square_in_place();

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

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
#include "functions/function_id.hpp"

/* TODO:
 *  Add templated overloads for member functions returning cumeric_t
 */

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

    struct AxisIndex
    {
        Axis axis;
        dim_t index;
    };

    class Tensor
    {
    public:
        /* Constructors */
        Tensor(const Shape& shape, datatype dtype = default_type, layout layout = layout::ANY); // FP32, ANY

        Tensor(cumeric_t value, datatype dtype = default_type, layout layout = layout::X); // scalar constructor
        Tensor(dim_t lenght, datatype dtype = default_type, layout layout = layout::X); // vector constructor
        Tensor(dim_t rows, dim_t cols, datatype dtype = default_type, layout layout = layout::NC); // Matrix Constructor
        Tensor(dim_t axis0, dim_t axis1, dim_t axis2, datatype dtype = default_type, layout layout = layout::ABC); // 3-rd rank tensor constructor
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

        static Tensor take_memory(const Shape& shape, void* data, datatype dtype = datatype::FP32, layout layout = layout::ANY);

        static Tensor make_cube(dim_t width, dim_t height, dim_t deepth, datatype dtype = default_type, layout layout = layout::OI);
        static Tensor make_matrix(dim_t rows, dim_t cols, datatype dtype = default_type, layout layout = layout::OI);
        static Tensor make_vector(dim_t lenght, datatype dtype = default_type, layout layout = layout::X);
        static Tensor make_scalar(datatype dtype = default_type, layout layout = layout::X);

        static Tensor Random(const Shape& shape, cumeric_t min = -1_c, cumeric_t max = 1_c, datatype dtype = datatype::FP32, layout layout = layout::ANY);
        static Tensor Zeros(const Shape& shape, datatype dtype = datatype::FP32, layout layout = layout::ANY);
        static Tensor Ones(const Shape& shape, datatype dtype = datatype::FP32, layout layout = layout::ANY);
        static Tensor Linspace(cumeric_t start, cumeric_t end, dim_t num); // vector

        /* Memory */

        Tensor& cast(datatype dtype); // In place

        Tensor& prefetch();

        Tensor& to_host();

        Tensor& to_device();

        /* getters */

        dim_t size() const;
        dim_t rank() const;
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
            return static_cast<const T*>(data())[idx];
        }



        const void* data() const; // I should add templated data getter
        void* data();



        template<typename T>
        T* data() { return static_cast<T*>(data()); }

        template<typename T>
        const T* data() const { return static_cast<const T*>(data()); }

        cumeric_t at(const Shape& indices);
        cumeric_t at(const Shape& indices) const;


        cumeric_t& operator () (const Shape& indices);
        const cumeric_t& operator () (const Shape& indices) const;
        
        cumeric_t operator ()(dim_t row, dim_t col) const;
        cumeric_t operator ()(dim_t idx0, dim_t idx1, dim_t idx2) const;
        cumeric_t operator ()(dim_t idx0, dim_t idx1, dim_t idx2, dim_t idx3) const;
        cumeric_t operator ()(dim_t idx0, dim_t idx1, dim_t idx2, dim_t idx3, dim_t idx4) const;

        Tensor batch(dim_t index) const;
        Tensor channel(dim_t index) const;
        Tensor row(dim_t index) const;
        Tensor col(dim_t index) const;

        /* Reshaping */
        Tensor slice(const Shape& offset, const Shape& shape) const;

        Tensor reshape(const Shape& shape) const;
        Tensor& reshape_in_place(const Shape& shape);

        Tensor transpose();
        Tensor& transpose_in_place();

        /* reductions */

        cumeric_t sum();
        cumeric_t mean();
        cumeric_t amean();

        cumeric_t squaredNorm();
        cumeric_t squared_norm();

        Tensor colwise_sum();
        Tensor rowwise_sum();
        Tensor channelwise_sum(); // for 3D channel-first tensors

        /* elementwise */

        Tensor elementwise(functions::function_id function) const;
        Tensor& elementwise_in_place(functions::function_id function);

        Tensor elementwise_diff(functions::function_id function) const;
        Tensor& elementwise_diff_in_place(functions::function_id function);

        Tensor& fill(cumeric_t scalar);

        Tensor multiply(const Tensor& tensor);
        Tensor cwiseProduct(const Tensor& tensor);

        Tensor& cwise_product_in_place(const Tensor& other);

        Tensor sqrt() const;
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
        Tensor(neural_primitives::Descriptor&& desc, const neural_primitives::Memory& source);
        dim_t compute_index(const Shape& indices) const;
        void* compute_address(const Shape& indices);

        std::unique_ptr<neural_primitives::Descriptor> _desc_;
        std::unique_ptr<neural_primitives::Memory> _memr_;
    };
} // cum

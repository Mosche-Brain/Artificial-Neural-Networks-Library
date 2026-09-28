//
// Created by jaro on 6/28/26.

#pragma once

#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <valarray>

#include "Matrix.hpp"
#include "cum/Core.hpp"
#include "cum/memory.hpp"
#include "cum/neural_primitives/Descriptor.hpp"
#include "cum/neural_primitives/Memory.hpp"
#include "cum/neural_primitives/opaque_types.hpp"
#include "functions/function_id.hpp"

/* TODO:
 *  Add templated overloads for member functions returning cumeric_t
 */

namespace cum
{
    // Inefficient container for storing references for arbitrary datatypes nonspecified in compile time
    // std::variant_alternative<>

    template<typename T>
    struct datatype_of;

    class Tensor
    {
    public:
        /* Constructors */
        Tensor(const Shape& shape, datatype dtype = default_type, layout layout = layout::ANY); // FP32, ANY
        Tensor() = default;

        Tensor(cumeric_t value, datatype dtype = default_type, layout layout = layout::A); // scalar constructor
        Tensor(dim_t lenght, datatype dtype = default_type, layout layout = layout::A); // vector constructor
        Tensor(dim_t rows, dim_t cols, datatype dtype = default_type, layout layout = layout::AB); // Matrix Constructor
        Tensor(dim_t axis0, dim_t axis1, dim_t axis2, datatype dtype = default_type, layout layout = layout::ABC); // 3-rd rank tensor constructor
        Tensor(dim_t axis0, dim_t axis1, dim_t axis2, dim_t axis3, datatype dtype = default_type, layout layout = layout::ABCD); // 4-th rank tensor constructor
        Tensor(dim_t axis0, dim_t axis1, dim_t axis2, dim_t axis3, dim_t axis4, datatype dtype = default_type, layout layout = layout::ABCDE); // 5-th rank tensor constructor

        Tensor(const Tensor& tensor);
        Tensor(Tensor&& tensor) noexcept;

        template<typename T, dim_t... dims> // Eigen3-style tensor constructor
        Tensor(const std::vector<T>& values);

        ~Tensor();

        /* fabriques */

        template<datatype T, layout layout>
        static Tensor create_tensor(const Shape& shape);

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

        bool is_scalar() const;
        // bool is_vector();
        // bool is_matrix();
        // bool has(Axis axis) const;
        // dim_t extent(Axis axis) const;

        // dim_t batches() const;
        // dim_t channels() const;
        // dim_t depth() const;
        // dim_t height() const;
        // dim_t width() const;
        dim_t rows() const;    // alias height()
        dim_t cols() const;    // alias width()

        std::unique_ptr<neural_primitives::Descriptor>& descriptor();
        std::unique_ptr<neural_primitives::Memory>& memory();

        const std::unique_ptr<neural_primitives::Descriptor>& descriptor() const;
        const std::unique_ptr<neural_primitives::Memory>& memory() const;

        Tensor clone() const;

        /* accesors */

        template<typename T>
        T* data() { return static_cast<T*>(data()); }

        template<typename T>
        const T* data() const { return static_cast<const T*>(data()); }

        template<typename T>
        T& at(const Shape& indices);

        template<typename T>
        const T& at(const Shape& indices) const;

        const void* data() const; // I should add templated data getter
        void* data();

        cumeric_t& at(const Shape& indices);
        const cumeric_t& at(const Shape& indices) const;

        cumeric_t get_value(const Shape& indices) const;

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

        cumeric_t sum() const;
        cumeric_t mean() const;
        cumeric_t amean() const;
        cumeric_t norm() const;
        cumeric_t squaredNorm() const;
        cumeric_t squared_norm() const;

        Tensor colwise_sum();
        Tensor rowwise_sum();
        Tensor channelwise_sum(); // for 3D channel-first tensors

        /* elementwise */

        Tensor elementwise(functions::function_id function) const;
        Tensor& elementwise_in_place(functions::function_id function);

        Tensor elementwise_diff(functions::function_id function) const;
        Tensor& elementwise_diff_in_place(functions::function_id function);

        Tensor& fill(cumeric_t scalar);

        Tensor multiply(const Tensor& tensor) const;
        Tensor cwiseProduct(const Tensor& tensor) const;

        Tensor& cwise_product_in_place(const Tensor& other);

        Tensor sqrt() const;
        Tensor& sqrt_in_place();

        Tensor square() const;
        Tensor& square_in_place();

        Tensor abs() const;
        Tensor& abs_in_place();

        Tensor clamp(cumeric_t min, cumeric_t max) const;
        Tensor& clamp_in_place(cumeric_t min, cumeric_t max);

        Tensor log() const;
        Tensor& log_in_place();

        Tensor& scale(cumeric_t scalar);

        /* matmul */

        Tensor matmul(const Tensor& other) const;

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

        bool friend operator == (const Tensor& A, const Tensor& B);
        bool friend operator != (const Tensor& A, const Tensor& B);

        Tensor& operator = (const Tensor& other);
        Tensor& operator = (Tensor&& other) noexcept;
    private:
        Tensor(neural_primitives::Descriptor&& desc, const neural_primitives::Memory& source);
        dim_t compute_index(const Shape& indices) const;
        void* compute_address(const Shape& indices);

        std::unique_ptr<neural_primitives::Descriptor> _desc_;
        std::unique_ptr<neural_primitives::Memory> _memr_;
        void* _data = nullptr;
        bool _owns_data = true;
    };


    template<typename T, dim_t... dims>
    Tensor::Tensor(const std::vector<T>& values) : Tensor(Shape{dims...}, datatype_of<T>())
    {
        constexpr dim_t elements_count = (dims * ...);

        if (values.size() != elements_count)
            throw std::invalid_argument("Tensor constructor: PI(dims) != lenght of values");

        const T* data = values.data();
        cum::memory::memcopy(_data, data, elements_count * sizeof(T));
    }

    template<datatype T, layout layout>
    Tensor Tensor::create_tensor(const Shape& shape)
    {
        return {shape, T, layout};
    }

    template<typename T>
    T& Tensor::at(const Shape& indices)
    {
        if (sizeof(T) != datatype_size(this->type()))
            throw std::runtime_error("datatype size mismatch");

        return data<T>()[compute_index(indices)];
        // return *slice(indices, Shape(rank(), 1)).data<T>();
    }

    template<typename T>
    const T& Tensor::at(const Shape& indices) const
    {
        if (sizeof(T) != datatype_size(this->type()))
            throw std::runtime_error("datatype size mismatch");

        return data<T>()[compute_index(indices)];
        // return *slice(indices, Shape(rank(), 1)).data<T>();
    }

    // I will move these specializations to another header, problably cum/datatypes.hpp
    template<>
    struct datatype_of<double>
    {
        static constexpr datatype value = datatype::FP64;
    };

    template<>
    struct datatype_of<float>
    {
        static constexpr datatype value = datatype::FP32;
    };

    template<>
    struct datatype_of<cum::float16>
    {
        static constexpr datatype value = datatype::FP16;
    };

    template<>
    struct datatype_of<cum::bfloat16>
    {
        static constexpr datatype value = datatype::BF16;
    };

    template<>
    struct datatype_of<cum::float8>
    {
        static constexpr datatype value = datatype::FP8;
    };

    template<>
    struct datatype_of<std::int64_t>
    {
        static constexpr datatype value = datatype::S64;
    };

    template<>
    struct datatype_of<std::int32_t>
    {
        static constexpr datatype value = datatype::S32;
    };

    template<>
    struct datatype_of<std::int16_t>
    {
        static constexpr datatype value = datatype::S16;
    };

    template<>
    struct datatype_of<std::int8_t>
    {
        static constexpr datatype value = datatype::S8;
    };

    template<>
    struct datatype_of<std::uint64_t>
    {
        static constexpr datatype value = datatype::U64;
    };

    template<>
    struct datatype_of<std::uint32_t>
    {
        static constexpr datatype value = datatype::U32;
    };

    template<>
    struct datatype_of<std::uint16_t>
    {
        static constexpr datatype value = datatype::U16;
    };

    template<>
    struct datatype_of<std::uint8_t>
    {
        static constexpr datatype value = datatype::U8;
    };

} // cum

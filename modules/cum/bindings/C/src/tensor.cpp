/*
 * @author: jaro
 * @name:   tensor
 * @file:   modules/cum/bindings/C/src/tensor.cpp
 * @date:   15 September 2026 21:58:55
 */
#include <exception>
#include <new>
#include <string>

#include "cum/cum.h"
#include "cum/cum.hpp"
#include "cum/Tensor.hpp"
#include "cum/detail/opaques_impl.hpp"

extern "C"
{

    #define CUM_BINARY_OPERATION(name, operation) \
    cum_status_t name(cum_tensor_t* dst, const cum_tensor_t* lhs, const cum_tensor_t* rhs) \
    { \
        if(dst == nullptr || lhs == nullptr || rhs == nullptr) \
            return failure(CUM_STATUS_INVALID_ARGUMENT, "invalid tensor"); \
        try \
        { \
            dst->tensor = lhs->tensor + rhs->tensor; \
            last_error.clear(); \
            return CUM_STATUS_SUCCESS; \
        } \
        catch(const std::exception& error) \
        { \
            return failure(CUM_STATUS_RUNTIME_ERROR, error); \
        } \
    }

    CUM_BINARY_OPERATION(cum_tensor_add, add)
    CUM_BINARY_OPERATION(cum_tensor_sub, sub)
    CUM_BINARY_OPERATION(cum_tensor_mul, mul)
    CUM_BINARY_OPERATION(cum_tensor_div, div)
    CUM_BINARY_OPERATION(cum_tensor_mmul, matmul)

    #undef CUM_BINARY_OPERATION
    cum_status_t cum_tensor_create(
        cum_tensor_t** tensor,
        cum_shape_t shape,
        cum_datatype_t datatype,
        cum_layout_t layout
    )
    {
        if(tensor == nullptr || (shape.rank != 0 && shape.dims == nullptr))
            return failure(CUM_STATUS_INVALID_ARGUMENT, "invalid tensor or shape");

        try
        {
            cum::Shape cpp_shape;
            if(shape.rank != 0)
                cpp_shape.assign(shape.dims, shape.dims + shape.rank);

            *tensor = new cum_tensor_t {
                cum::Tensor(cpp_shape, to_datatype(datatype), to_layout(layout))
            };
            last_error.clear();
            return CUM_STATUS_SUCCESS;
        }
        catch(const std::bad_alloc& error)
        {
            return failure(CUM_STATUS_OUT_OF_MEMORY, error);
        }
        catch(const std::exception& error)
        {
            return failure(CUM_STATUS_RUNTIME_ERROR, error);
        }
    }

    cum_status_t cum_tensor_destroy(cum_tensor_t* tensor)
    {
        delete tensor;
        return CUM_STATUS_SUCCESS;
    }

    cum_status_t cum_tensor_copy(cum_tensor_t* dst, const cum_tensor_t* src)
    {
        if(dst == nullptr || src == nullptr)
            return failure(CUM_STATUS_INVALID_ARGUMENT, "invalid tensor");

        try
        {
            dst->tensor = src->tensor;
            last_error.clear();
            return CUM_STATUS_SUCCESS;
        }
        catch(const std::exception& error)
        {
            return failure(CUM_STATUS_RUNTIME_ERROR, error);
        }
    }

    cum_status_t cum_tensor_shape(
        const cum_tensor_t* tensor,
        dim_t* dims,
        size_t capacity,
        size_t* rank
    )
    {
        if(tensor == nullptr || rank == nullptr)
            return failure(CUM_STATUS_INVALID_ARGUMENT, "invalid tensor or rank");

        try
        {
            const cum::Shape shape = tensor->tensor.shape();
            *rank = shape.size();
            if(dims == nullptr)
                return CUM_STATUS_SUCCESS;
            if(capacity < shape.size())
                return failure(CUM_STATUS_INVALID_ARGUMENT, "shape buffer is too small");

            for(size_t index = 0; index < shape.size(); ++index)
                dims[index] = shape[index];

            last_error.clear();
            return CUM_STATUS_SUCCESS;
        }
        catch(const std::exception& error)
        {
            return failure(CUM_STATUS_RUNTIME_ERROR, error);
        }
    }

    cum_status_t cum_tensor_size(const cum_tensor_t* tensor, dim_t* size)
    {
        if(tensor == nullptr || size == nullptr)
            return failure(CUM_STATUS_INVALID_ARGUMENT, "invalid tensor or size");
        *size = tensor->tensor.size();
        return CUM_STATUS_SUCCESS;
    }

    cum_status_t cum_tensor_length(const cum_tensor_t* tensor, dim_t* length)
    {
        if(tensor == nullptr || length == nullptr)
            return failure(CUM_STATUS_INVALID_ARGUMENT, "invalid tensor or length");
        *length = tensor->tensor.lenght();
        return CUM_STATUS_SUCCESS;
    }

    cum_status_t cum_tensor_datatype(const cum_tensor_t* tensor, cum_datatype_t* datatype)
    {
        if(tensor == nullptr || datatype == nullptr)
            return failure(CUM_STATUS_INVALID_ARGUMENT, "invalid tensor or datatype");
        *datatype = static_cast<cum_datatype_t>(tensor->tensor.type());
        return CUM_STATUS_SUCCESS;
    }

    cum_status_t cum_tensor_layout(const cum_tensor_t* tensor, cum_layout_t* layout)
    {
        if(tensor == nullptr || layout == nullptr)
            return failure(CUM_STATUS_INVALID_ARGUMENT, "invalid tensor or layout");
        *layout = static_cast<cum_layout_t>(tensor->tensor.format());
        return CUM_STATUS_SUCCESS;
    }

    cum_status_t cum_tensor_data(const cum_tensor_t* tensor, const void** data)
    {
        if(tensor == nullptr || data == nullptr)
            return failure(CUM_STATUS_INVALID_ARGUMENT, "invalid tensor or data");
        *data = tensor->tensor.data();
        return CUM_STATUS_SUCCESS;
    }

    cum_status_t cum_tensor_fill(cum_tensor_t* tensor, cumeric_t scalar)
    {
        if(tensor == nullptr)
            return failure(CUM_STATUS_INVALID_ARGUMENT, "invalid tensor");
        try
        {
            tensor->tensor.fill(scalar);
            last_error.clear();
            return CUM_STATUS_SUCCESS;
        }
        catch(const std::exception& error)
        {
            return failure(CUM_STATUS_RUNTIME_ERROR, error);
        }
    }

    #define CUM_IN_PLACE_OPERATION(name, operation) \
    cum_status_t name(cum_tensor_t* lhs, const cum_tensor_t* rhs) \
    { \
        if(lhs == nullptr || rhs == nullptr) \
            return failure(CUM_STATUS_INVALID_ARGUMENT, "invalid tensor"); \
        try \
        { \
            lhs->tensor += rhs->tensor; \
            last_error.clear(); \
            return CUM_STATUS_SUCCESS; \
        } \
        catch(const std::exception& error) \
        { \
            return failure(CUM_STATUS_RUNTIME_ERROR, error); \
        } \
    }

    CUM_IN_PLACE_OPERATION(cum_tensor_add_in_place, add)
    CUM_IN_PLACE_OPERATION(cum_tensor_sub_in_place, sub)
    CUM_IN_PLACE_OPERATION(cum_tensor_mul_in_place, mul)
    CUM_IN_PLACE_OPERATION(cum_tensor_div_in_place, div)

    #undef CUM_IN_PLACE_OPERATION
}
#include "yann_internal.hpp"
#include "yann/utils/safetensors.hpp"

#include <exception>
#include <vector>

extern "C" {

static cum::layout default_layout_for_rank(size_t ndims)
{
    switch (ndims)
    {
        case 1:  return cum::layout::A;
        case 2:  return cum::layout::AB;
        case 3:  return cum::layout::ABC;
        case 4:  return cum::layout::ABCD;
        case 5:  return cum::layout::ABCDE;
        default: return cum::layout::ANY;
    }
}

yann_status_t yann_tensor_create_nd(
    yann_context_t* /*ctx*/,
    const int64_t* dims,
    size_t ndims,
    yann_tensor_t** out_tensor
)
{
    if (!out_tensor || (!dims && ndims > 0) || ndims == 0)
    {
        yann_set_error("yann_tensor_create_nd: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        cum::Shape shape(dims, dims + ndims);
        auto* t = new yann_tensor_s{};
        t->tensor = cum::Tensor::Zeros(shape, cum::default_type, default_layout_for_rank(ndims));
        *out_tensor = t;
        return YANN_OK;
    }
    catch (const std::bad_alloc& e)
    {
        yann_set_error(std::string("yann_tensor_create_nd out of memory: ") + e.what());
        return YANN_ERROR_OUT_OF_MEMORY;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_tensor_create_nd failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_tensor_create(
    yann_context_t* ctx,
    size_t rows,
    size_t cols,
    yann_tensor_t** out_tensor
)
{
    const int64_t dims[2] = {static_cast<int64_t>(rows), static_cast<int64_t>(cols)};
    return yann_tensor_create_nd(ctx, dims, 2, out_tensor);
}

yann_status_t yann_tensor_copy_from_host(
    yann_tensor_t* tensor,
    const float* src,
    size_t src_len
)
{
    if (!tensor || !src)
    {
        yann_set_error("yann_tensor_copy_from_host: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        const size_t total = static_cast<size_t>(tensor->tensor.lenght());
        if (src_len < total)
        {
            yann_set_error("yann_tensor_copy_from_host: source buffer smaller than tensor length");
            return YANN_ERROR_INVALID_ARGUMENT;
        }

        cum::cumeric_t* dst = tensor->tensor.data<cum::cumeric_t>();
        for (size_t i = 0; i < total; ++i)
        {
            dst[i] = static_cast<cum::cumeric_t>(src[i]);
        }
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_tensor_copy_from_host failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_tensor_create_nd_from_host(
    yann_context_t* ctx,
    const int64_t* dims,
    size_t ndims,
    const float* host_data,
    yann_tensor_t** out_tensor
)
{
    if (!host_data)
    {
        yann_set_error("yann_tensor_create_nd_from_host: host_data is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    yann_tensor_t* t = nullptr;
    yann_status_t status = yann_tensor_create_nd(ctx, dims, ndims, &t);
    if (status != YANN_OK)
        return status;

    const size_t len = static_cast<size_t>(t->tensor.lenght());
    status = yann_tensor_copy_from_host(t, host_data, len);
    if (status != YANN_OK)
    {
        yann_tensor_destroy(t);
        return status;
    }

    *out_tensor = t;
    return YANN_OK;
}

yann_status_t yann_tensor_create_from_host(
    yann_context_t* ctx,
    size_t rows,
    size_t cols,
    const float* host_data,
    yann_tensor_t** out_tensor
)
{
    const int64_t dims[2] = {static_cast<int64_t>(rows), static_cast<int64_t>(cols)};
    return yann_tensor_create_nd_from_host(ctx, dims, 2, host_data, out_tensor);
}

yann_status_t yann_tensor_copy_to_host(
    const yann_tensor_t* tensor,
    float* dst,
    size_t dst_len
)
{
    if (!tensor || !dst)
    {
        yann_set_error("yann_tensor_copy_to_host: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        const size_t total = static_cast<size_t>(tensor->tensor.lenght());
        if (dst_len < total)
        {
            yann_set_error("yann_tensor_copy_to_host: destination buffer too small");
            return YANN_ERROR_INVALID_ARGUMENT;
        }

        const cum::cumeric_t* src = tensor->tensor.data<cum::cumeric_t>();
        for (size_t i = 0; i < total; ++i)
        {
            dst[i] = static_cast<float>(src[i]);
        }
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_tensor_copy_to_host failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_tensor_fill(yann_tensor_t* tensor, float value)
{
    if (!tensor)
    {
        yann_set_error("yann_tensor_fill: tensor is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        tensor->tensor.fill(static_cast<cum::cumeric_t>(value));
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_tensor_fill failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_tensor_get_2d(
    const yann_tensor_t* tensor,
    size_t row,
    size_t col,
    float* out_value
)
{
    if (!tensor || !out_value)
    {
        yann_set_error("yann_tensor_get_2d: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        *out_value = static_cast<float>(
            tensor->tensor.at<cum::cumeric_t>({static_cast<cum::dim_t>(row), static_cast<cum::dim_t>(col)})
        );
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_tensor_get_2d failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_tensor_set_2d(
    yann_tensor_t* tensor,
    size_t row,
    size_t col,
    float value
)
{
    if (!tensor)
    {
        yann_set_error("yann_tensor_set_2d: tensor is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        tensor->tensor.at<cum::cumeric_t>({static_cast<cum::dim_t>(row), static_cast<cum::dim_t>(col)}) =
            static_cast<cum::cumeric_t>(value);
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_tensor_set_2d failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_tensor_get_shape(
    const yann_tensor_t* tensor,
    size_t* out_rows,
    size_t* out_cols
)
{
    if (!tensor || !out_rows || !out_cols)
    {
        yann_set_error("yann_tensor_get_shape: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        const cum::Shape shape = tensor->tensor.shape();
        *out_rows = shape.size() >= 1 ? static_cast<size_t>(shape[0]) : 0;
        *out_cols = shape.size() >= 2 ? static_cast<size_t>(shape[1]) : 1;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_tensor_get_shape failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

size_t yann_tensor_get_rank(const yann_tensor_t* tensor)
{
    if (!tensor)
        return 0;
    return static_cast<size_t>(tensor->tensor.rank());
}

size_t yann_tensor_get_length(const yann_tensor_t* tensor)
{
    if (!tensor)
        return 0;
    return static_cast<size_t>(tensor->tensor.lenght());
}

yann_status_t yann_tensor_get_dims(
    const yann_tensor_t* tensor,
    int64_t* out_dims,
    size_t max_dims
)
{
    if (!tensor || !out_dims)
    {
        yann_set_error("yann_tensor_get_dims: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    const cum::Shape shape = tensor->tensor.shape();
    if (max_dims < shape.size())
    {
        yann_set_error("yann_tensor_get_dims: max_dims buffer too small");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    for (size_t i = 0; i < shape.size(); ++i)
        out_dims[i] = shape[i];

    return YANN_OK;
}

yann_status_t yann_tensor_save_safetensors(
    const yann_tensor_t* tensor,
    const char* filename,
    const char* name
)
{
    if (!tensor || !filename)
    {
        yann_set_error("yann_tensor_save_safetensors: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        const std::string key = name ? name : "tensor";
        yann::utils::save_safetensors({{key, tensor->tensor}}, filename);
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_tensor_save_safetensors failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_tensor_load_safetensors(
    const char* filename,
    const char* name,
    yann_tensor_t** out_tensor
)
{
    if (!filename || !out_tensor)
    {
        yann_set_error("yann_tensor_load_safetensors: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        auto map = yann::utils::load_safetensors(filename);
        const std::string key = name ? name : "tensor";
        auto it = map.find(key);
        if (it == map.end())
        {
            yann_set_error("yann_tensor_load_safetensors: key '" + key + "' not found");
            return YANN_ERROR_INVALID_ARGUMENT;
        }

        auto* t = new yann_tensor_s{it->second};
        *out_tensor = t;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_tensor_load_safetensors failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

void yann_tensor_destroy(yann_tensor_t* tensor)
{
    delete tensor;
}

} // extern "C"

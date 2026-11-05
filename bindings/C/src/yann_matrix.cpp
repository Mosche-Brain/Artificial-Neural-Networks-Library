#include "yann_internal.hpp"
#include <exception>

extern "C" {

/* Legacy matrix aliases forwarding directly to yann_tensor_* */

yann_status_t yann_matrix_create(
    yann_context_t* ctx,
    size_t rows,
    size_t cols,
    yann_matrix_t** out_tensor
)
{
    return yann_tensor_create(ctx, rows, cols, out_tensor);
}

yann_status_t yann_matrix_create_from_host(
    yann_context_t* ctx,
    size_t rows,
    size_t cols,
    const float* host_data,
    yann_matrix_t** out_tensor
)
{
    return yann_tensor_create_from_host(ctx, rows, cols, host_data, out_tensor);
}

yann_status_t yann_matrix_copy_to_host(
    const yann_matrix_t* tensor,
    float* dst,
    size_t dst_len
)
{
    return yann_tensor_copy_to_host(tensor, dst, dst_len);
}

yann_status_t yann_matrix_get_shape(
    const yann_matrix_t* tensor,
    size_t* out_rows,
    size_t* out_cols
)
{
    return yann_tensor_get_shape(tensor, out_rows, out_cols);
}

void yann_matrix_destroy(yann_matrix_t* tensor)
{
    yann_tensor_destroy(tensor);
}

/* --------------------------- Parameter API -------------------------------- */

yann_status_t yann_parameter_create_zeros(size_t rows, size_t cols, yann_parameter_t** out_param)
{
    if (!out_param)
    {
        yann_set_error("yann_parameter_create_zeros: out_param is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        auto* p = new yann_parameter_s{yann::Parameter::Zeros(static_cast<cum::dim_t>(rows), static_cast<cum::dim_t>(cols))};
        *out_param = p;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_parameter_create_zeros failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_parameter_create_ones(size_t rows, size_t cols, yann_parameter_t** out_param)
{
    if (!out_param)
    {
        yann_set_error("yann_parameter_create_ones: out_param is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        auto* p = new yann_parameter_s{yann::Parameter::Ones(static_cast<cum::dim_t>(rows), static_cast<cum::dim_t>(cols))};
        *out_param = p;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_parameter_create_ones failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_parameter_create_uniform(size_t rows, size_t cols, yann_parameter_t** out_param)
{
    if (!out_param)
    {
        yann_set_error("yann_parameter_create_uniform: out_param is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        auto* p = new yann_parameter_s{yann::Parameter::Uniform(static_cast<cum::dim_t>(rows), static_cast<cum::dim_t>(cols))};
        *out_param = p;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_parameter_create_uniform failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_parameter_create_he(size_t rows, size_t cols, yann_parameter_t** out_param)
{
    if (!out_param)
    {
        yann_set_error("yann_parameter_create_he: out_param is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        auto* p = new yann_parameter_s{yann::Parameter::He(static_cast<cum::dim_t>(rows), static_cast<cum::dim_t>(cols))};
        *out_param = p;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_parameter_create_he failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_parameter_create_xavier(size_t rows, size_t cols, yann_parameter_t** out_param)
{
    if (!out_param)
    {
        yann_set_error("yann_parameter_create_xavier: out_param is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        auto* p = new yann_parameter_s{yann::Parameter::Xavier(static_cast<cum::dim_t>(rows), static_cast<cum::dim_t>(cols))};
        *out_param = p;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_parameter_create_xavier failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_parameter_load_safetensors(
    const char* filename,
    const char* name,
    bool load_gradient,
    yann_parameter_t** out_param
)
{
    if (!filename || !out_param)
    {
        yann_set_error("yann_parameter_load_safetensors: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        const char* key = name ? name : "values";
        auto* p = new yann_parameter_s{yann::Parameter::FromSafetensors(filename, key, load_gradient)};
        *out_param = p;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_parameter_load_safetensors failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_parameter_save_safetensors(
    const yann_parameter_t* param,
    const char* filename,
    const char* name,
    bool include_gradient
)
{
    if (!param || !filename)
    {
        yann_set_error("yann_parameter_save_safetensors: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        const char* key = name ? name : "values";
        param->param.save_safetensors(filename, include_gradient, key);
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_parameter_save_safetensors failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_parameter_get_values(const yann_parameter_t* param, yann_tensor_t** out_tensor)
{
    if (!param || !out_tensor)
    {
        yann_set_error("yann_parameter_get_values: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        *out_tensor = new yann_tensor_s{param->param.values};
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_parameter_get_values failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_parameter_get_gradient(const yann_parameter_t* param, yann_tensor_t** out_tensor)
{
    if (!param || !out_tensor)
    {
        yann_set_error("yann_parameter_get_gradient: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        *out_tensor = new yann_tensor_s{param->param.gradient};
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_parameter_get_gradient failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_parameter_clear_gradient(yann_parameter_t* param)
{
    if (!param)
    {
        yann_set_error("yann_parameter_clear_gradient: param is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        param->param.clear_gradient();
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_parameter_clear_gradient failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

void yann_parameter_destroy(yann_parameter_t* param)
{
    delete param;
}

} // extern "C"

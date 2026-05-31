#include "yann_internal.hpp"

#include "Models/Layers/Dense.hpp"
#include "Models/Layers/LayerBase.hpp"

#include <memory>
#include <stdexcept>

extern "C" {

yann_status_t yann_model_create(yann_context_t ctx, yann_model_t* out_model)
{
    if (!ctx || !out_model)
    {
        yann_set_error("yann_model_create: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        auto* m = new yann_model_t{};
        // Sequential is default-constructed as empty
        *out_model = m;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_create failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

void yann_model_destroy(yann_model_t model)
{
    if (model)
    {
        delete model;
    }
}

yann_status_t yann_model_add_dense(yann_model_t model,
                                   size_t units,
                                   const char* activation)
{
    if (!model || !activation)
    {
        yann_set_error("yann_model_add_dense: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        // Use the factory method if available, otherwise construct directly
        auto layer = YANN::Models::Layers::Dense::createUnique(
            static_cast<int>(units),
            activation
        );

        if (!layer)
        {
            yann_set_error("yann_model_add_dense: failed to create Dense layer");
            return YANN_ERROR_RUNTIME;
        }

        model->model.addLayer(std::move(layer));
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_add_dense failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_model_set_input_size(yann_model_t model, size_t input_size)
{
    if (!model)
    {
        yann_set_error("yann_model_set_input_size: model is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    // In the current YANN design, input size is usually inferred from the first forward call.
    // We can store it for later validation or explicit initialization if needed.
    // For now we just accept it (future-proofing).
    (void)input_size;
    return YANN_OK;
}

yann_status_t yann_model_predict(yann_model_t model,
                                 yann_tensor_t input,
                                 yann_tensor_t* output)
{
    if (!model || !input || !output)
    {
        yann_set_error("yann_model_predict: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        // Run forward pass
        cum::Matrix result = model->model.forward(input->matrix);

        // Wrap the result into a new tensor that the caller owns
        auto* out_tensor = new yann_tensor_t{};
        out_tensor->matrix = std::move(result);   // move to avoid extra copy when possible

        *output = out_tensor;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_predict failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
    catch (...)
    {
        yann_set_error("yann_model_predict: unknown error during forward pass");
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_model_fit(yann_model_t model,
                             yann_tensor_t X,
                             yann_tensor_t Y,
                             float learning_rate,
                             size_t epochs)
{
    if (!model || !X || !Y)
    {
        yann_set_error("yann_model_fit: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        model->model.fit(X->matrix, Y->matrix,
                         static_cast<cum::cumeric_t>(learning_rate),
                         epochs);
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_fit failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

size_t yann_model_get_layer_count(yann_model_t model)
{
    if (!model)
        return 0;
    return model->model.getLayersCount();
}

yann_status_t yann_model_get_layer_output_size(yann_model_t model,
                                               size_t layer_index,
                                               size_t* out_size)
{
    if (!model || !out_size)
    {
        yann_set_error("yann_model_get_layer_output_size: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    if (layer_index >= model->model.getLayersCount())
    {
        yann_set_error("yann_model_get_layer_output_size: layer index out of range");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        // We need a way to query layer size. Currently LayerBase has virtual int size()
        auto layer = model->model.getLayer(layer_index); // Note: this returns by value in current API
        // This is problematic because getLayer currently does almost nothing.

        // Workaround: use the last output size of the model if we are at the last layer,
        // otherwise we need to expose more from LayerBase.
        // For now we return the size of the last layer's output as a best effort.

        if (layer_index == model->model.getLayersCount() - 1)
        {
            // Return number of columns in the last output (common pattern)
            // This is not perfect because we don't have direct access here.
            *out_size = 0; // Placeholder - real implementation needs better introspection
            return YANN_ERROR_NOT_IMPLEMENTED;
        }

        *out_size = 0;
        return YANN_ERROR_NOT_IMPLEMENTED;
    }
    catch (...)
    {
        return YANN_ERROR_RUNTIME;
    }
}

} // extern "C"

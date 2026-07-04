#include "yann_internal.hpp"

#include "Models/Layers/Dense.hpp"

extern "C" {

yann_status_t yann_model_create(yann_context_t* ctx, yann_model_t** out_model)
{
    if (!out_model) {
        yann_set_error("yann_model_create: out_model is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try {
        *out_model = new yann_model_s{};
        return YANN_OK;
    } catch (const std::exception& e) {
        yann_set_error(std::string("yann_model_create failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

void yann_model_destroy(yann_model_t* model)
{
    delete model;
}

yann_status_t yann_model_add_dense(yann_model_t* model,
                                   size_t units,
                                   const char* activation)
{
    if (!model || !activation) {
        yann_set_error("yann_model_add_dense: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto layer = yann::models::layers::Dense::createUnique(
            static_cast<int>(units), activation);

        if (!layer) {
            yann_set_error("yann_model_add_dense: failed to create layer");
            return YANN_ERROR_RUNTIME;
        }

        model->model.addLayer(std::move(layer));
        return YANN_OK;
    } catch (const std::exception& e) {
        yann_set_error(std::string("yann_model_add_dense failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_model_predict(yann_model_t* model,
                                 yann_matrix_t* input,
                                 yann_matrix_t** output)
{
    if (!model || !input || !output) {
        yann_set_error("yann_model_predict: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try {
        cum::Matrix result = model->model.forward(input->matrix);

        auto* out_tensor = new yann_matrix_s{};
        out_tensor->matrix = std::move(result);
        *output = out_tensor;

        return YANN_OK;
    } catch (const std::exception& e) {
        yann_set_error(std::string("yann_model_predict failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_model_fit(yann_model_t* model,
                             yann_matrix_t* X,
                             yann_matrix_t* Y,
                             float learning_rate,
                             size_t epochs)
{
    if (!model || !X || !Y) {
        yann_set_error("yann_model_fit: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try {
        model->model.fit(X->matrix, Y->matrix,
                         static_cast<cum::cumeric_t>(learning_rate),
                         epochs);
        return YANN_OK;
    } catch (const std::exception& e) {
        yann_set_error(std::string("yann_model_fit failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

size_t yann_model_get_layer_count(yann_model_t* model)
{
    if (!model) return 0;
    return model->model.getLayersCount();
}

} // extern "C"
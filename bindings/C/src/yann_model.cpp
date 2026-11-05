#include "yann_internal.hpp"

#include "yann/loss.hpp"
#include "yann/models/layers/Layers.hpp"
#include "yann/optimizers.hpp"
#include "yann/utils/serialize.hpp"

#include <exception>

extern "C" {

/* --------------------------- Optimizers ----------------------------------- */

yann_status_t yann_optimizer_create_sgd(float learning_rate, yann_optimizer_t** out_opt)
{
    if (!out_opt)
    {
        yann_set_error("yann_optimizer_create_sgd: out_opt is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        auto* o = new yann_optimizer_s{};
        o->optimizer = yann::optimizers::SGD::create(static_cast<cum::cumeric_t>(learning_rate));
        *out_opt = o;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_optimizer_create_sgd failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_optimizer_create_adam(float learning_rate, float b1, float b2, yann_optimizer_t** out_opt)
{
    if (!out_opt)
    {
        yann_set_error("yann_optimizer_create_adam: out_opt is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        auto* o = new yann_optimizer_s{};
        o->optimizer = yann::optimizers::ADAM::create(
            static_cast<cum::cumeric_t>(learning_rate),
            static_cast<cum::cumeric_t>(b1),
            static_cast<cum::cumeric_t>(b2)
        );
        *out_opt = o;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_optimizer_create_adam failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_optimizer_create_momentum(float learning_rate, float beta, yann_optimizer_t** out_opt)
{
    if (!out_opt)
    {
        yann_set_error("yann_optimizer_create_momentum: out_opt is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        auto* o = new yann_optimizer_s{};
        o->optimizer = yann::optimizers::Momentum::create(
            static_cast<cum::cumeric_t>(learning_rate),
            static_cast<cum::cumeric_t>(beta)
        );
        *out_opt = o;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_optimizer_create_momentum failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_optimizer_create_adagrad(float learning_rate, float eps, yann_optimizer_t** out_opt)
{
    if (!out_opt)
    {
        yann_set_error("yann_optimizer_create_adagrad: out_opt is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        auto* o = new yann_optimizer_s{};
        o->optimizer = yann::optimizers::AdaGrad::create(
            static_cast<cum::cumeric_t>(learning_rate),
            static_cast<cum::cumeric_t>(eps)
        );
        *out_opt = o;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_optimizer_create_adagrad failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

void yann_optimizer_destroy(yann_optimizer_t* opt)
{
    delete opt;
}

/* --------------------------- Loss Functions ------------------------------- */

yann_status_t yann_loss_create_mse(yann_loss_t** out_loss)
{
    if (!out_loss)
    {
        yann_set_error("yann_loss_create_mse: out_loss is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        auto* l = new yann_loss_s{};
        l->loss = yann::loss::MeanSquaredError::create();
        *out_loss = l;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_loss_create_mse failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_loss_create_mae(yann_loss_t** out_loss)
{
    if (!out_loss)
    {
        yann_set_error("yann_loss_create_mae: out_loss is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        auto* l = new yann_loss_s{};
        l->loss = yann::loss::MeanAbsoluteError::create();
        *out_loss = l;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_loss_create_mae failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_loss_create_bce(yann_loss_t** out_loss)
{
    if (!out_loss)
    {
        yann_set_error("yann_loss_create_bce: out_loss is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        auto* l = new yann_loss_s{};
        l->loss = yann::loss::BinaryCrossEntropy::create();
        *out_loss = l;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_loss_create_bce failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_loss_create_cce(int from_logits, yann_loss_t** out_loss)
{
    if (!out_loss)
    {
        yann_set_error("yann_loss_create_cce: out_loss is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        auto* l = new yann_loss_s{};
        l->loss = yann::loss::CategoricalCrossEntropy::create(from_logits != 0);
        *out_loss = l;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_loss_create_cce failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_loss_compute(
    yann_loss_t* loss,
    const yann_tensor_t* predicted,
    const yann_tensor_t* target,
    float* out_value
)
{
    if (!loss || !predicted || !target || !out_value)
    {
        yann_set_error("yann_loss_compute: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        loss->loss->compute(predicted->tensor, target->tensor);
        *out_value = static_cast<float>(loss->loss->getLoss());
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_loss_compute failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

void yann_loss_destroy(yann_loss_t* loss)
{
    delete loss;
}

/* --------------------------- Sequential Model ----------------------------- */

yann_status_t yann_model_create(yann_context_t* /*ctx*/, yann_model_t** out_model)
{
    if (!out_model)
    {
        yann_set_error("yann_model_create: out_model is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        *out_model = new yann_model_s{};
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_create failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

void yann_model_destroy(yann_model_t* model)
{
    delete model;
}

yann_status_t yann_model_add_input(yann_model_t* model, size_t features)
{
    if (!model)
    {
        yann_set_error("yann_model_add_input: model is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        model->model.addLayer(yann::models::layers::Input::createUnique(static_cast<int>(features)));
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_add_input failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_model_add_input_nd(yann_model_t* model, const int64_t* dims, size_t ndims)
{
    if (!model || !dims || ndims == 0)
    {
        yann_set_error("yann_model_add_input_nd: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        cum::Shape shape(dims, dims + ndims);
        model->model.addLayer(yann::models::layers::Input::createUnique(shape));
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_add_input_nd failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_model_add_dense(yann_model_t* model, size_t units, const char* activation)
{
    if (!model || !activation)
    {
        yann_set_error("yann_model_add_dense: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        auto layer = yann::models::layers::Dense::createUnique(static_cast<int>(units), activation);
        model->model.addLayer(std::move(layer));
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_add_dense failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_model_add_linear(yann_model_t* model, size_t units)
{
    if (!model)
    {
        yann_set_error("yann_model_add_linear: model is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        model->model.addLayer(yann::models::layers::Linear::createUnique(static_cast<int>(units)));
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_add_linear failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_model_add_flatten(yann_model_t* model)
{
    if (!model)
    {
        yann_set_error("yann_model_add_flatten: model is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        model->model.addLayer(yann::models::layers::Flatten::createUnique());
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_add_flatten failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_model_add_softmax(yann_model_t* model, size_t length, int64_t axis)
{
    if (!model)
    {
        yann_set_error("yann_model_add_softmax: model is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        model->model.addLayer(yann::models::layers::Softmax::createUnique(static_cast<cum::dim_t>(length), axis));
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_add_softmax failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_model_build(yann_model_t* model, const int64_t* input_dims, size_t ndims)
{
    if (!model)
    {
        yann_set_error("yann_model_build: model is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        cum::Shape shape;
        if (input_dims && ndims > 0)
            shape.assign(input_dims, input_dims + ndims);
        model->model.build(shape);
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_build failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_model_forward(
    yann_model_t* model,
    const yann_tensor_t* input,
    yann_tensor_t** output
)
{
    if (!model || !input || !output)
    {
        yann_set_error("yann_model_forward: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        if (model->model.getLayersCount() > 0 && !model->model.getTopology().front()->initialized())
        {
            const cum::Shape in_shape = input->tensor.shape();
            cum::Shape feature_shape;
            if (in_shape.size() > 1)
                feature_shape.assign(in_shape.begin() + 1, in_shape.end());
            else
                feature_shape = in_shape;
            model->model.build(feature_shape);
        }

        cum::Tensor result = model->model.forward(input->tensor);
        *output = new yann_tensor_s{std::move(result)};
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_forward failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_model_predict(
    yann_model_t* model,
    const yann_tensor_t* input,
    yann_tensor_t** output
)
{
    return yann_model_forward(model, input, output);
}

yann_status_t yann_model_backward(
    yann_model_t* model,
    const yann_tensor_t* d_output,
    yann_tensor_t** d_input
)
{
    if (!model || !d_output)
    {
        yann_set_error("yann_model_backward: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        cum::Tensor grad_in = model->model.backward(d_output->tensor);
        if (d_input)
            *d_input = new yann_tensor_s{std::move(grad_in)};
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_backward failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_model_fit_ex(
    yann_model_t* model,
    const yann_tensor_t* X,
    const yann_tensor_t* Y,
    yann_loss_t* loss,
    yann_optimizer_t* optimizer,
    size_t epochs,
    size_t batch_size
)
{
    if (!model || !X || !Y || !loss || !optimizer)
    {
        yann_set_error("yann_model_fit_ex: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        if (model->model.getLayersCount() > 0 && !model->model.getTopology().front()->initialized())
        {
            const cum::Shape in_shape = X->tensor.shape();
            cum::Shape feature_shape;
            if (in_shape.size() > 1)
                feature_shape.assign(in_shape.begin() + 1, in_shape.end());
            else
                feature_shape = in_shape;
            model->model.build(feature_shape);
        }

        model->model.fit(
            X->tensor,
            Y->tensor,
            *loss->loss,
            *optimizer->optimizer,
            static_cast<cum::dim_t>(epochs),
            static_cast<cum::dim_t>(batch_size)
        );
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_fit_ex failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_model_fit(
    yann_model_t* model,
    const yann_tensor_t* X,
    const yann_tensor_t* Y,
    float learning_rate,
    size_t epochs
)
{
    if (!model || !X || !Y)
    {
        yann_set_error("yann_model_fit: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        auto loss = yann::loss::MeanSquaredError::create();
        auto opt  = yann::optimizers::SGD::create(static_cast<cum::cumeric_t>(learning_rate));

        if (model->model.getLayersCount() > 0 && !model->model.getTopology().front()->initialized())
        {
            const cum::Shape in_shape = X->tensor.shape();
            cum::Shape feature_shape;
            if (in_shape.size() > 1)
                feature_shape.assign(in_shape.begin() + 1, in_shape.end());
            else
                feature_shape = in_shape;
            model->model.build(feature_shape);
        }

        model->model.fit(X->tensor, Y->tensor, *loss, *opt, static_cast<cum::dim_t>(epochs), 1);
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_fit failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_model_save_safetensors(const yann_model_t* model, const char* filename)
{
    if (!model || !filename)
    {
        yann_set_error("yann_model_save_safetensors: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        yann::utils::model_serialize_to_safetensors(model->model, filename);
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_save_safetensors failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_model_load_safetensors(yann_model_t* model, const char* filename)
{
    if (!model || !filename)
    {
        yann_set_error("yann_model_load_safetensors: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        yann::utils::model_deserialize_from_safetensors(model->model, filename);
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_model_load_safetensors failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

size_t yann_model_get_layer_count(const yann_model_t* model)
{
    if (!model)
        return 0;
    return model->model.getLayersCount();
}

/* --------------------------- Chain View ----------------------------------- */

yann_status_t yann_chain_create(
    yann_model_t* const* models,
    size_t model_count,
    yann_chain_t** out_chain
)
{
    if (!out_chain || (!models && model_count > 0))
    {
        yann_set_error("yann_chain_create: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        auto* c = new yann_chain_s{};
        for (size_t i = 0; i < model_count; ++i)
        {
            if (!models[i])
            {
                delete c;
                yann_set_error("yann_chain_create: null model pointer in array");
                return YANN_ERROR_INVALID_ARGUMENT;
            }
            c->chain.addModel(models[i]->model);
        }
        *out_chain = c;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_chain_create failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_chain_add_model(yann_chain_t* chain, yann_model_t* model)
{
    if (!chain || !model)
    {
        yann_set_error("yann_chain_add_model: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        chain->chain.addModel(model->model);
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_chain_add_model failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_chain_build(yann_chain_t* chain, const int64_t* input_dims, size_t ndims)
{
    if (!chain)
    {
        yann_set_error("yann_chain_build: chain is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        cum::Shape shape;
        if (input_dims && ndims > 0)
            shape.assign(input_dims, input_dims + ndims);
        chain->chain.build(shape);
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_chain_build failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_chain_forward(
    yann_chain_t* chain,
    const yann_tensor_t* input,
    yann_tensor_t** output
)
{
    if (!chain || !input || !output)
    {
        yann_set_error("yann_chain_forward: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        cum::Tensor result = chain->chain.forward(input->tensor);
        *output = new yann_tensor_s{std::move(result)};
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_chain_forward failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_chain_backward(
    yann_chain_t* chain,
    const yann_tensor_t* d_output,
    yann_tensor_t** d_input
)
{
    if (!chain || !d_output)
    {
        yann_set_error("yann_chain_backward: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        cum::Tensor grad_in = chain->chain.backward(d_output->tensor);
        if (d_input)
            *d_input = new yann_tensor_s{std::move(grad_in)};
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_chain_backward failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_chain_fit(
    yann_chain_t* chain,
    const yann_tensor_t* X,
    const yann_tensor_t* Y,
    yann_loss_t* loss,
    yann_optimizer_t* optimizer,
    size_t epochs,
    size_t batch_size
)
{
    if (!chain || !X || !Y || !loss || !optimizer)
    {
        yann_set_error("yann_chain_fit: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        chain->chain.fit(
            X->tensor,
            Y->tensor,
            *loss->loss,
            *optimizer->optimizer,
            static_cast<cum::dim_t>(epochs),
            static_cast<cum::dim_t>(batch_size)
        );
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_chain_fit failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

size_t yann_chain_get_model_count(const yann_chain_t* chain)
{
    if (!chain)
        return 0;
    return chain->chain.size();
}

void yann_chain_destroy(yann_chain_t* chain)
{
    delete chain;
}

} // extern "C"

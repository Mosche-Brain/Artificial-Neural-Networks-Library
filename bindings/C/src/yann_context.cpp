#include "yann_internal.hpp"
#include <exception>

extern "C" {

static cum::DEVICE to_cum_device(yann_device_t device)
{
    switch (device)
    {
        case YANN_DEVICE_CPU:  return cum::DEVICE::CPU;
        case YANN_DEVICE_GPU:  return cum::DEVICE::GPU;
        case YANN_DEVICE_AUTO:
        default:               return cum::DEVICE::AUTO;
    }
}

yann_status_t yann_create_context_with_device(yann_device_t device, yann_context_t** out_ctx)
{
    if (!out_ctx)
    {
        yann_set_error("yann_create_context_with_device: out_ctx is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        const cum::DEVICE cum_dev = to_cum_device(device);
        cum::cum(cum_dev);
        auto* ctx = new yann_context_s{};
        ctx->device = cum_dev;
        *out_ctx = ctx;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("Failed to create context: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_create_context(yann_context_t** out_ctx)
{
    return yann_create_context_with_device(YANN_DEVICE_AUTO, out_ctx);
}

void yann_destroy_context(yann_context_t* ctx)
{
    if (!ctx)
        return;
    try
    {
        cum::decum();
    }
    catch (...)
    {
    }
    delete ctx;
}

} // extern "C"

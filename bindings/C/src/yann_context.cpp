#include "yann_internal.hpp"

#include <exception>

extern "C" {

yann_status_t yann_create_context(yann_context_t* out_ctx)
{
    if (!out_ctx)
    {
        yann_set_error("yann_create_context: out_ctx is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        *out_ctx = new yann_context_t{};
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("Failed to create context: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
    catch (...)
    {
        yann_set_error("Failed to create context: unknown error");
        return YANN_ERROR_RUNTIME;
    }
}

void yann_destroy_context(yann_context_t ctx)
{
    if (ctx)
    {
        delete ctx;
    }
}

} // extern "C"

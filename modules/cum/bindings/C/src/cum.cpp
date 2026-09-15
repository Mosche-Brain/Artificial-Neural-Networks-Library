#include <exception>
#include <new>
#include <string>

#include "cum/cum.h"
#include "cum/cum.hpp"
#include "cum/Tensor.hpp"
#include "cum/detail/opaques_impl.hpp"

namespace
{
    thread_local std::string last_error;

    cum_status_t failure(cum_status_t status, const std::exception& error)
    {
        last_error = error.what();
        return status;
    }

    cum_status_t failure(cum_status_t status, const char* error)
    {
        last_error = error;
        return status;
    }

    cum::datatype to_datatype(cum_datatype_t datatype)
    {
        return static_cast<cum::datatype>(datatype);
    }

    cum::layout to_layout(cum_layout_t layout)
    {
        return static_cast<cum::layout>(layout);
    }
}

extern "C"
{
    cum_status_t cum_init(cum_device_t device)
    {
        try
        {
            cum::cum(static_cast<cum::DEVICE>(device));
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

    cum_status_t cum_shutdown(void)
    {
        try
        {
            cum::decum();
            last_error.clear();
            return CUM_STATUS_SUCCESS;
        }
        catch(const std::exception& error)
        {
            return failure(CUM_STATUS_RUNTIME_ERROR, error);
        }
    }

    const char* cum_last_error(void)
    {
        return last_error.c_str();
    }




}

#include "yann_internal.hpp"
#include "yann/runtime_config.hpp"

extern "C" {

const char* yann_get_last_error(void)
{
    return g_last_error.c_str();
}

void yann_set_verbosity(unsigned char level)
{
    yann::runtime_config::set_verbosity(level);
}

} // extern "C"

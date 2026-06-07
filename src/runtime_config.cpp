#include "runtime_config.hpp"

namespace YANN::runtime_config
{
    int DEBUG_VEBOSITY = 1; // every + 1 is one more indentation level in debug output, 0 means no debug output, 1 means only layer level output, 2 means layer and operation level output, and so on

    unsigned int verbosity_level() 
    { 
        return RuntimeConfig::getInstance().getDebugVerbosity(); 
    }

    void set_verbosity(unsigned int level)
    {
        RuntimeConfig::getInstance().setDebugVerbosity(level); 
    }
}

namespace YANN
{
    static RuntimeConfig* runtime_config_instance = nullptr;

    RuntimeConfig& RuntimeConfig::getInstance()
    {
        if(!runtime_config_instance)
        {
            runtime_config_instance = new RuntimeConfig();
        }
        return *runtime_config_instance;
    }
} // namespace YANN

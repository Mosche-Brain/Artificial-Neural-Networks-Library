#include "runtime_config.hpp"

namespace yann::runtime_config
{
    unsigned char verbosity_level()
    { 
        return logging::Logger::getInstance().getDebugVerbosity();
    }

    bool fused_kernels()
    {
        return RuntimeConfig::getInstance().getFusedKernels();
    }

    void set_verbosity(unsigned char level)
    {
        logging::Logger::getInstance().setDebugVerbosity(level);
    }
}

namespace yann
{
    static RuntimeConfig* runtime_config_instance = nullptr;

    RuntimeConfig::RuntimeConfig() : async_mode(false), fused_kernels(false), cached_preactivations(true), enable_telemetry(true)
    {

    };

    RuntimeConfig& RuntimeConfig::getInstance()
    {
        if(!runtime_config_instance)
        {
            runtime_config_instance = new RuntimeConfig();
        }
        return *runtime_config_instance;
    }

    void RuntimeConfig::setAsyncMode(bool mode)
    {
        async_mode = mode;
    }

    bool RuntimeConfig::getAsyncMode() const
    {
        return async_mode;
    }

    void RuntimeConfig::setFusedKernels(bool mode)
    {
        fused_kernels = mode;
    }

    bool RuntimeConfig::getFusedKernels() const
    {
        return fused_kernels;
    }

    void RuntimeConfig::setCachedPreactivations(bool mode)
    {
        cached_preactivations = mode;
    }

    bool RuntimeConfig::getCachedPreactivations() const
    {
        return cached_preactivations;
    }

    void RuntimeConfig::setIsTelemetryEnabled(bool enable)
    {
        enable_telemetry = enable;
    }

    bool RuntimeConfig::getIsTelemetryEnabled() const
    {
        return enable_telemetry;
    }

    logging::Logger& logger()
    {
        return logging::Logger::getInstance();
    }

} // namespace YANN

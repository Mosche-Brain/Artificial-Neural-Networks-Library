#pragma once

#include <stdint.h>
#include "logging/Logger.hpp"

namespace yann::runtime_config
{
    unsigned char verbosity_level();

    /*
     * 1 - Training Level
     * 2 - Epoch level
     * 3 - Sample level
     * 4 - Layer level
     * 5 - Operation level
     */
    void set_verbosity(unsigned char level);

    bool async_mode();
    void set_async_mode(bool mode);

    bool fused_kernels();
    void set_fused_kernels(bool fused_kernels);

    bool cached_preactivations();
    void set_cached_preactivations(bool cached_preactivations);

    bool telemetry_enabled();
    void enable_telemetry(bool enable);

}

namespace yann
{
    class RuntimeConfig    
    {   
    private:
        bool async_mode;
        bool fused_kernels;
        bool cached_preactivations;
        bool enable_telemetry;

        // utils::Logger logger;

        RuntimeConfig();
    public:
        RuntimeConfig(const RuntimeConfig&) = delete;
        RuntimeConfig& operator = (const RuntimeConfig&) = delete;

        void setAsyncMode(bool mode);
        bool getAsyncMode() const;

        void setFusedKernels(bool mode);
        bool getFusedKernels() const;

        void setCachedPreactivations(bool mode);
        bool getCachedPreactivations() const;

        void setIsTelemetryEnabled(bool enable);
        bool getIsTelemetryEnabled() const;

        static RuntimeConfig& getInstance();
    };

    logging::Logger& logger();
}
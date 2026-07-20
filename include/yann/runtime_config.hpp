#pragma once

#include <stdint.h>

namespace yann::runtime_config
{
    extern int DEBUG_VEBOSITY; // every + 1 is one more indentation level in debug output, 0 means no debug output, 1 means only layer level output, 2 means layer and operation level output, and so on

    bool cached_preactivations();
    void set_cached_preactivations(bool cached_preactivations);

    bool fused_kernels();
    void set_fused_kernels(bool fused_kernels);

    int8_t verbosity_level();
    void set_verbosity(unsigned int level);
}

namespace yann
{
    class RuntimeConfig    
    {   
    private:
        int8_t debug_verbosity; // every + 1 is one more indentation level in debug output, 0 means no debug output, 1 means only layer level output, 2 means layer and operation level output, and so on
        bool async_mode;
        bool fused_kernels;
        bool cached_preactivations;
        bool enable_telemetry;

        RuntimeConfig() : debug_verbosity(1), enable_telemetry(true) {};
    public:
        RuntimeConfig(const RuntimeConfig&) = delete;
        RuntimeConfig& operator = (const RuntimeConfig&) = delete;

        void setDebugVerbosity(int verbosity) { debug_verbosity = verbosity; }
        unsigned char getDebugVerbosity()     { return debug_verbosity; }

        void setIsTelemetryEnabled(bool enable) { enable_telemetry = enable; }
        bool getIsTelemetryEnabled()            { return enable_telemetry; }

        static RuntimeConfig& getInstance();
    };
}
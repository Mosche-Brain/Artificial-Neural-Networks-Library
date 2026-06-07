#pragma once

namespace YANN::runtime_config
{
    extern int DEBUG_VEBOSITY; // every + 1 is one more indentation level in debug output, 0 means no debug output, 1 means only layer level output, 2 means layer and operation level output, and so on
    
    unsigned int verbosity_level();
    void set_verbosity(unsigned int level);
}

namespace YANN
{
    class RuntimeConfig    
    {   
    private:
        int debug_verbosity; // every + 1 is one more indentation level in debug output, 0 means no debug output, 1 means only layer level output, 2 means layer and operation level output, and so on
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
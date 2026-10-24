/**
 * @file runtime_config.hpp
 * @brief Global runtime configuration management and options for YANN.
 */

#pragma once

#include <stdint.h>
#include "logging/Logger.hpp"

namespace yann::runtime_config
{
    /**
     * @brief Get the current runtime logging verbosity level.
     * @return Verbosity level as an unsigned character.
     */
    unsigned char verbosity_level();

    /**
     * @brief Set the global logging verbosity level.
     *
     * Levels:
     * - 1: Training level
     * - 2: Epoch level
     * - 3: Batch level
     * - 4: Layer level
     * - 5: Operation level
     *
     * @param level Verbosity level to set.
     */
    void set_verbosity(unsigned char level);

    /**
     * @brief Check whether asynchronous execution mode is currently enabled.
     * @return True if asynchronous mode is active, false otherwise.
     */
    bool async_mode();

    /**
     * @brief Enable or disable asynchronous compute mode.
     * @param mode Desired asynchronous execution state.
     */
    void set_async_mode(bool mode);

    /**
     * @brief Check if fused kernel execution is enabled.
     * @return True if fused kernels are enabled, false otherwise.
     */
    bool fused_kernels();

    /**
     * @brief Enable or disable fused kernel execution.
     * @param fused_kernels Desired fused kernels state.
     */
    void set_fused_kernels(bool fused_kernels);

    /**
     * @brief Check whether pre-activation caching is enabled.
     * @return True if pre-activations are cached, false otherwise.
     */
    bool cached_preactivations();

    /**
     * @brief Enable or disable caching of pre-activation values during forward passes.
     * @param cached_preactivations Desired caching state.
     */
    void set_cached_preactivations(bool cached_preactivations);

    /**
     * @brief Check if runtime telemetry is enabled.
     * @return True if telemetry is enabled, false otherwise.
     */
    bool telemetry_enabled();

    /**
     * @brief Enable or disable runtime performance and diagnostics telemetry.
     * @param enable Desired telemetry state.
     */
    void enable_telemetry(bool enable);

}

namespace yann
{
    /**
     * @class RuntimeConfig
     * @brief Singleton class holding global execution parameters and runtime toggles.
     */
    class RuntimeConfig    
    {   
    private:
        bool async_mode;
        bool fused_kernels;
        bool cached_preactivations;
        bool enable_telemetry;
        // cum::dim_t batch_axis;

        RuntimeConfig();
    public:
        RuntimeConfig(const RuntimeConfig&) = delete;
        RuntimeConfig& operator = (const RuntimeConfig&) = delete;

        /**
         * @brief Set asynchronous compute execution mode.
         * @param mode Boolean flag indicating if async execution is active.
         */
        void setAsyncMode(bool mode);

        /**
         * @brief Check whether asynchronous compute execution is active.
         * @return Current async execution mode state.
         */
        bool getAsyncMode() const;

        /**
         * @brief Enable or disable fused computational kernels.
         * @param mode Desired fused kernel execution state.
         */
        void setFusedKernels(bool mode);

        /**
         * @brief Get status of fused kernel execution.
         * @return True if fused kernels are used, false otherwise.
         */
        bool getFusedKernels() const;

        /**
         * @brief Set whether pre-activations should be kept in cache.
         * @param mode Desired caching state.
         */
        void setCachedPreactivations(bool mode);

        /**
         * @brief Query pre-activation caching status.
         * @return True if pre-activations are cached, false otherwise.
         */
        bool getCachedPreactivations() const;

        /**
         * @brief Enable or disable diagnostics and telemetry tracking.
         * @param enable Desired telemetry tracking state.
         */
        void setIsTelemetryEnabled(bool enable);

        /**
         * @brief Query telemetry state.
         * @return True if telemetry is currently enabled.
         */
        bool getIsTelemetryEnabled() const;

        /**
         * @brief Get reference to the singleton RuntimeConfig instance.
         * @return Reference to the global RuntimeConfig instance.
         */
        static RuntimeConfig& getInstance();
    };

    /**
     * @brief Access the global default logging instance.
     * @return Reference to the shared logging::Logger instance.
     */
    logging::Logger& logger();
}

/**
 * @file Logger.hpp
 * @author jaro
 * @brief Thread-safe logging facility and macros for debugging and monitoring YANN.
 * @date 2026-07-24
 */

#ifndef YANN_LOGGER_HPP
#define YANN_LOGGER_HPP

#include <string>
#include <format>

#if defined(ENABLE_DEBUG_OUTPUT)
/**
 * @def YANN_LOG
 * @brief Macro to log formatted messages at a specific verbosity level if ENABLE_DEBUG_OUTPUT is defined.
 */
#define YANN_LOG(level, fmt, ...) yann::logging::Logger::getInstance().log(level, std::format(fmt, __VA_ARGS__))
// #define YANN_LOG(level, fmt) yann::utils::Logger::getInstance().log(level, fmt)
#else
/**
 * @def YANN_LOG
 * @brief Disabled log macro when ENABLE_DEBUG_OUTPUT is not defined.
 */
#define YANN_LOG(level, fmt, ...)
// #define YANN_LOG(level, fmt)
#endif

namespace yann::logging
{
    /**
     * @class Logger
     * @brief Singleton logger class managing message output based on verbosity levels.
     */
    class Logger
    {
    public:
        /**
         * @brief Get reference to the singleton Logger instance.
         * @return Reference to Logger instance.
         */
        static Logger& getInstance();

        /**
         * @brief Set maximum logging verbosity threshold.
         * @param level Verbosity level integer.
         */
        void setDebugVerbosity(unsigned char level);

        /**
         * @brief Get current logging verbosity threshold.
         * @return Current verbosity level integer.
         */
        unsigned char getDebugVerbosity() const;

        // template<typename... Args>
        /**
         * @brief Log a string message if verbose_level does not exceed the logger threshold.
         * @param verbose_level Importance level of the message.
         * @param fmt Log message string.
         */
        void log(unsigned char verbose_level, const std::string& fmt) const; // verbose_level tells in wchich verbosity level the message should be logged
        // void log(unsigned char verbose_level, std::format_string<char> fmt,  auto&&... args);
        // void log(unsigned char verbose_level, std::format_string<Args...> fmt,  Args&&... args);
        // void debug(unsigned char verbose_level, std::format_string<Args...> fmt,  Args&&... args);
        // void info(unsigned char verbose_level, std::format_string<Args...> fmt,  Args&&... args);
        // void error(unsigned char verbose_level, std::format_string<Args...> fmt,  Args&&... args);
    private:
        Logger() = default;
        unsigned char verbosity_level = 1;
    };
} // yann::logging

#endif //YANN_LOGGER_HPP

//
// Created by jaro on 7/24/26.
//

#ifndef YANN_LOGGER_HPP
#define YANN_LOGGER_HPP


#include <string>
#include <format>

#if defined(ENABLE_DEBUG_OUTPUT)
#define YANN_LOG(level, fmt, ...) yann::logging::Logger::getInstance().log(level, std::format(fmt, __VA_ARGS__))
// #define YANN_LOG(level, fmt) yann::utils::Logger::getInstance().log(level, fmt)
#else
#define YANN_LOG(level, fmt, ...)
// #define YANN_LOG(level, fmt)
#endif

namespace yann::logging
{

    class Logger
    {
    public:
        static Logger& getInstance();

        void setDebugVerbosity(unsigned char level);
        unsigned char getDebugVerbosity() const;

        // template<typename... Args>
        void log(unsigned char verbose_level, const std::string& fmt) const; // verbose_level tells in wchich verbosity level the message should be logged
        // void log(unsigned char verbose_level, std::format_string<char> fmt,  auto&&... args); // verbose_level tells in wchich verbosity level the message should be logged
        // void log(unsigned char verbose_level, std::format_string<Args...> fmt,  Args&&... args); // verbose_level tells in wchich verbosity level the message should be logged
        // void debug(unsigned char verbose_level, std::format_string<Args...> fmt,  Args&&... args); // verbose_level tells in wchich verbosity level the message should be logged
        // void info(unsigned char verbose_level, std::format_string<Args...> fmt,  Args&&... args); // verbose_level tells in wchich verbosity level the message should be logged
        // void error(unsigned char verbose_level, std::format_string<Args...> fmt,  Args&&... args); // verbose_level tells in wchich verbosity level the message should be logged
    private:
        Logger() = default;
        unsigned char verbosity_level = 1;
    };
} // yann::utils

#endif //YANN_LOGGER_HPP

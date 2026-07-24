//
// Created by jaro on 7/24/26.
//

#ifndef YANN_LOGGER_HPP
#define YANN_LOGGER_HPP

#include <string>
#include <format>

namespace yann::utils
{

    class Logger
    {
    public:
        static Logger& getInstance();

        void setDebugVerbosity(unsigned char level);
        unsigned char getDebugVerbosity() const;

        // template<typename... Args>
        void log(unsigned char verbose_level, std::format_string<char> fmt,  auto&&... args); // verbose_level tells in wchich verbosity level the message should be logged
        // void log(unsigned char verbose_level, std::format_string<Args...> fmt,  Args&&... args); // verbose_level tells in wchich verbosity level the message should be logged
        // void debug(unsigned char verbose_level, std::format_string<Args...> fmt,  Args&&... args); // verbose_level tells in wchich verbosity level the message should be logged
        // void info(unsigned char verbose_level, std::format_string<Args...> fmt,  Args&&... args); // verbose_level tells in wchich verbosity level the message should be logged
        // void error(unsigned char verbose_level, std::format_string<Args...> fmt,  Args&&... args); // verbose_level tells in wchich verbosity level the message should be logged
    private:
        unsigned char verbosity_level;
    };
} // yann::utils

#endif //YANN_LOGGER_HPP

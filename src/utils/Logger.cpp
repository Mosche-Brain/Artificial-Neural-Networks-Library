//
// Created by jaro on 7/24/26.
//

#include "yann/utils/Logger.hpp"

#include <print>
#include <utility>

namespace yann::utils
{
    static Logger* logger_instance = nullptr;

    Logger& Logger::getInstance()
    {
        if(!logger_instance)
        {
            logger_instance = new Logger();
        }
        return *logger_instance;
    }

    void Logger::setDebugVerbosity(unsigned char level)
    {
        verbosity_level = level;
    }

    unsigned char Logger::getDebugVerbosity() const
    {
        return verbosity_level;
    }

    inline void Logger::log(unsigned char verbose_level, std::format_string<char> fmt,  auto&&... args)
    {
        if (verbose_level >= verbosity_level)
            std::print(fmt, std::forward<decltype(args)>(args)...);
    }
} // yann
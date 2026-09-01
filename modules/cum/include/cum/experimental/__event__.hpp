/**----------------------------------------------
 * ?                    ABOUT
 * @author      : Swiadomy-Mężczyzna
 * @email       : contact@mosche.dev
 * @event       : __event__.hpp
 * @createdOn   : 9/1/26
 * @description : Declaration of opaque __event__ structure
 *---------------------------------------------**/

#pragma once

#include "cum/Core.hpp"
#include <memory>

namespace cum::experimental 
{
    // struct __event__;

    class __event__
    {
    public:
        // __event__() = default;
        // __event__(__event__&& other) noexcept;
        // __event__& operator=(__event__&& other) noexcept;
        __event__(__event__&&) noexcept;
        __event__& operator=(__event__&&) noexcept;

        __event__(const __event__&) = delete;
        __event__& operator=(const __event__&) = delete;

        ~__event__();

        void wait();
    // private:

        struct Impl;
        
        explicit __event__(Impl* handle) noexcept;
        
        // Impl* handle = nullptr;
        std::unique_ptr<Impl> handle;
    };
}
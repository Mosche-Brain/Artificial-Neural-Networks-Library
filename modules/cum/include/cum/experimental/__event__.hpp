/**----------------------------------------------
 * ?                    ABOUT
 * @author      : Swiadomy-Mężczyzna
 * @email       : contact@mosche.dev
 * @event       : __event__.hpp
 * @createdOn   : 9/1/26
 * @description : Declaration of opaque __event__ structure
 *---------------------------------------------**/

#pragma once

#include <memory>

namespace cum::detail
{
    class make_event;
}

namespace cum::experimental 
{

    class __event__
    {
    public:
        // __event__() = default;
        // __event__(__event__&& other) noexcept;
        // __event__& operator=(__event__&& other) noexcept;
        __event__() noexcept;
        __event__(__event__&&) noexcept;
        __event__& operator=(__event__&&) noexcept;

        __event__(const __event__&) = delete;
        __event__& operator=(const __event__&) = delete;

        ~__event__();

        void wait();
         
    private:
        struct Impl;
        
        std::unique_ptr<Impl> handle;

        explicit __event__(std::unique_ptr<Impl> handle) noexcept;

        friend class cum::detail::make_event;
    };
}
/**----------------------------------------------
 * ?                    ABOUT
 * @author      : Swiadomy-Mężczyzna
 * @email       : contact@mosche.dev
 * @event       : __event__.hpp
 * @createdOn   : 9/1/26
 * @description : Declaration of opaque __event__ structure
 *---------------------------------------------**/

#pragma once

namespace cum::experimental 
{
    struct __event__;

    void wait_for_event(__event__& event);
}
//
// Created by jaro on 7/19/26.
//

#ifndef YANN_RUNTIME_HPP
#define YANN_RUNTIME_HPP

namespace cum::runtime
{
    // opaque queue structure
    typedef struct Device Device;
    typedef struct Queue Queue;

    void sync();
}

#endif //YANN_RUNTIME_HPP

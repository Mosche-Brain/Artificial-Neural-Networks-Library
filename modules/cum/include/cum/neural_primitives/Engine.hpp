//
// Created by jaro on 9/10/26.
//

#ifndef YANN_ENGINE_HPP
#define YANN_ENGINE_HPP

#include <memory>

#include "cum/neural_primitives/opaque_types.hpp"

#include "cum/Core.hpp"

namespace cum::neural_primitives
{
    class Engine
    {
    public:
        Engine();
        // Engine();

    private:
        friend class Memory;

        std::unique_ptr<handles::__engine__> handle;
    };
} // cum

#endif //YANN_ENGINE_HPP

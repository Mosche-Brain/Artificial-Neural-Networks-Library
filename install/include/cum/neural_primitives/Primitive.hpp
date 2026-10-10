/*
 * @author: jaro
 * @name:   Primitive
 * @file:   modules/cum/include/cum/neural_primitives/Primitive.hpp
 * @date:   19 September 2026 18:44:28
 */

#ifndef YANN_PRIMITIVE_HPP
#define YANN_PRIMITIVE_HPP

#include "cum/Core.hpp"

namespace cum::neural_primitives
{
    class Primitive
    {
    public:
        Primitive();

        static Primitive binary();
        static Primitive eltwise();
        static Primitive matmul();

        Primitive(const Primitive& other);

        __event__ execute();

        enum class Kind { binary, elementwise, matmul, softmax, sum, reduction, reorder };
    };
} // cum

#endif //YANN_PRIMITIVE_HPP

/*
 * @author: jaro
 * @name:   opaques_impl
 * @file:   modules/cum/bindings/C/include/cum/detail/opaques_impl.hpp
 * @date:   13 September 2026 22:47:40
 */

#ifndef YANN_OPAQUES_IMPL_HPP
#define YANN_OPAQUES_IMPL_HPP
#include "cum/Core.hpp"

namespace cum
{
    class Tensor;
    class Matrix;
    class Vector;

    namespace neural_primitives
    {
        class Engine;
        class Memory;
        class Descriptor;
    }
}
// Why am I forward declaring this? I can just include headers

typedef struct cum_tensor_t
{
    cum::Tensor tensor;
} cum_tensor_t;

typedef struct cum_datatype_t
{
    // cum::datatype
    // Wait, it could be just enum
} cum_datatype_t;

typedef struct
{
    cum::neural_primitives::Descriptor* descriptor;
} cum_descriptor_t;

typedef struct
{
    cum::neural_primitives::Memory* memory;
} cum_memory_t;

typedef struct
{
    cum::Shape shape;
} cum_shape_t;

typedef struct cum_layout_t
{
    // This also could be an enum
} cum_layout_t;

typedef struct
{
    cum::Tensor tensor;
} cum_tensor_t;

typedef struct
{
    cum::Matrix matrix;
} cum_matrix_t;

#endif //YANN_OPAQUES_IMPL_HPP

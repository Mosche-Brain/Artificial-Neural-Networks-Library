//
// Created by jaro on 7/19/26.
//

#include "cum/detail/vendor/oneapi/opaque_types.hpp"

#include "internal/context.hpp"

#include "cum/neural_primitives/Memory.hpp"

namespace cum::neural_primitives
{
    Memory::Memory(const Descriptor& desc)
    {
        handle = std::make_unique<handles::__memory__>(
            dnnl::memory{desc.handle->desc, internal::engine()}
        );
    }

    Memory::Memory(const Descriptor& desc, void* data)
    {
        handle = std::make_unique<handles::__memory__>(
            dnnl::sycl_interop::make_memory(desc.handle->desc, internal::engine(), dnnl::sycl_interop::memory_kind::usm, data)
        );
    }

    Memory::Memory(const Descriptor& desc, Engine& engine)
    {
        handle = std::make_unique<handles::__memory__>(
            dnnl::memory{desc.handle->desc, engine.handle->engine}
        );
    }

    Memory::Memory(const Descriptor& desc, Engine& engine, void* data)
    {

    }

    void* Memory::data()
    {
        return handle->memory.get_data_handle();
    }

    void Memory::set_data(void* data)
    {
        handle->memory.set_data_handle(data);
    }

}
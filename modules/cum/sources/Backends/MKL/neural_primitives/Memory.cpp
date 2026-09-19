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
        handle_ = std::make_unique<handles::__memory__>(
            dnnl::memory{desc.handle_->desc, internal::engine()}
        );
    }

    Memory::Memory(const Descriptor& desc, void* data)
    {
        handle_ = std::make_unique<handles::__memory__>(
            dnnl::sycl_interop::make_memory(desc.handle_->desc, internal::engine(), dnnl::sycl_interop::memory_kind::usm, data)
        );
    }

    Memory::Memory(const Descriptor& desc, Engine& engine)
    {
        handle_ = std::make_unique<handles::__memory__>(
            dnnl::memory{desc.handle_->desc, engine.handle->engine}
        );
    }

    Memory::Memory(handles::__memory__& handle)
    {
        handle_ = std::make_unique<handles::__memory__>(handle);
    }


    Memory::Memory(const Descriptor& desc, Engine& engine, void* data)
    {
        handle_ = std::make_unique<handles::__memory__>(
            dnnl::sycl_interop::make_memory(desc.handle_->desc, engine.handle->engine, dnnl::sycl_interop::memory_kind::usm, data)
        );
    }

    Memory::Memory(const Descriptor& desc, const Memory& source) : owner_(source.owner())
    {
        void* data = owner_->memory.get_data_handle();
        handle_ = std::make_unique<handles::__memory__>(
            dnnl::sycl_interop::make_memory(desc.handle_->desc, internal::engine(), dnnl::sycl_interop::memory_kind::usm, data)
        );
    }


    Memory::~Memory() = default;

    void* Memory::data()
    {
        return handle_->memory.get_data_handle();
    }

    const void* Memory::data() const
    {
        return handle_->memory.get_data_handle();
    }

    void Memory::set_data(void* data)
    {
        handle_->memory.set_data_handle(data);
    }

    const handles::__memory__& Memory::handle() const
    {
        return *handle_;
    }

    handles::__memory__& Memory::handle()
    {
        return *handle_;
    }

    std::shared_ptr<handles::__memory__> Memory::owner() const
    {
        return owner_ ? owner_ : handle_;
    }

    // const Descriptor& Memory::descriptor() const
    // {
    //     return D(handle_->memory.get_desc());
    // }

}
/**----------------------------------------------
 * ?                    ABOUT
 * @author      : Świadomy Mężczyzna
 * @email       : 
 * @file        : modules/cum/sources/Backends/MKL/neural_primitives/tensor_operations.hpp
 * @createdOn   : 09/12/26 
 * @description : ND tensors operation declarations
 *---------------------------------------------**/

#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/event.hpp>

#include "internal/context.hpp"

#include "cum/detail/vendor/oneapi/event_handler.hpp"
#include "cum/detail/vendor/oneapi/opaque_types.hpp"
#include "cum/Tensor.hpp"

#include "cum/neural_primitives/tensor_operations.hpp"

namespace cum::neural_primitives
{
    /* Raw handles overloads */
    __event__ add(handles::__memory__& C, const handles::__memory__& A, const handles::__memory__& B, const handles::__desc__& c_desc, const handles::__desc__& a_desc, const handles::__desc__& b_desc)
    {
	    dnnl::binary::primitive_desc primitive_desc {
			internal::engine(),
	    	dnnl::algorithm::binary_add,
	    	a_desc.desc,
	    	b_desc.desc,
	    	b_desc.desc,
	    };

    	dnnl::primitive primitive = dnnl::binary(primitive_desc);

        sycl::event event = dnnl::sycl_interop::execute(
            primitive, 
            internal::stream(), 
            {
				{ DNNL_ARG_SRC_0, A.memory },
				{ DNNL_ARG_SRC_1, B.memory },
				{ DNNL_ARG_DST, C.memory }
            }
        );


    	internal::stream().wait();
        return detail::event_handler::create(std::move(event));
    }

    __event__ sub(handles::__memory__& C, const handles::__memory__& A, const handles::__memory__& B, const handles::__desc__& c_desc, const handles::__desc__& a_desc, const handles::__desc__& b_desc)
    {
	    dnnl::binary::primitive_desc primitive_desc {
			internal::engine(),
	    	dnnl::algorithm::binary_sub,
	    	a_desc.desc,
	    	b_desc.desc,
	    	b_desc.desc,
	    };

    	dnnl::primitive primitive = dnnl::binary(primitive_desc);

        sycl::event event = dnnl::sycl_interop::execute(
            primitive, 
            internal::stream(), 
            {
				{ DNNL_ARG_SRC_0, A.memory },
				{ DNNL_ARG_SRC_1, B.memory },
				{ DNNL_ARG_DST, C.memory }
            }
        );


    	internal::stream().wait();
        return detail::event_handler::create(std::move(event));
    }

    __event__ mul(handles::__memory__& C, const handles::__memory__& A, const handles::__memory__& B, const handles::__desc__& c_desc, const handles::__desc__& a_desc, const handles::__desc__& b_desc)
    {
	    dnnl::binary::primitive_desc primitive_desc {
			internal::engine(),
	    	dnnl::algorithm::binary_mul,
	    	a_desc.desc,
	    	b_desc.desc,
	    	b_desc.desc,
	    };

    	dnnl::primitive primitive = dnnl::binary(primitive_desc);

        sycl::event event = dnnl::sycl_interop::execute(
            primitive, 
            internal::stream(), 
            {
				{ DNNL_ARG_SRC_0, A.memory },
				{ DNNL_ARG_SRC_1, B.memory },
				{ DNNL_ARG_DST, C.memory }
            }
        );


    	internal::stream().wait();
        return detail::event_handler::create(std::move(event));
    }

    __event__ div(handles::__memory__& C, const handles::__memory__& A, const handles::__memory__& B, const handles::__desc__& c_desc, const handles::__desc__& a_desc, const handles::__desc__& b_desc)
    {
	    dnnl::binary::primitive_desc primitive_desc {
			internal::engine(),
	    	dnnl::algorithm::binary_div,
	    	a_desc.desc,
	    	b_desc.desc,
	    	b_desc.desc,
	    };

    	dnnl::primitive primitive = dnnl::binary(primitive_desc);

        sycl::event event = dnnl::sycl_interop::execute(
            primitive, 
            internal::stream(), 
            {
				{ DNNL_ARG_SRC_0, A.memory },
				{ DNNL_ARG_SRC_1, B.memory },
				{ DNNL_ARG_DST, C.memory }
            }
        );


    	internal::stream().wait();
        return detail::event_handler::create(std::move(event));
    }

    
    __event__ matmul(handles::__memory__& C, const handles::__memory__& A, const handles::__memory__& B, const handles::__desc__& c_desc, const handles::__desc__& a_desc, const handles::__desc__& b_desc)
    {
	    dnnl::matmul::primitive_desc primitive_desc {
			internal::engine(),
	    	a_desc.desc,
	    	b_desc.desc,
	    	b_desc.desc,
	    };

    	dnnl::primitive primitive = dnnl::matmul(primitive_desc);

        sycl::event event = dnnl::sycl_interop::execute(
            primitive, 
            internal::stream(), 
            {
				{ DNNL_ARG_SRC_0, A.memory },
				{ DNNL_ARG_SRC_1, B.memory },
				{ DNNL_ARG_DST, C.memory }
            }
        );


    	internal::stream().wait();
        return detail::event_handler::create(std::move(event));
    }

    __event__ add(handles::__memory__& A, const handles::__memory__& B, const handles::__desc__& a_desc, const handles::__desc__& b_desc)
    {
	    return add(A, A, B, a_desc, a_desc, b_desc);
    }

    __event__ sub(handles::__memory__& A, const handles::__memory__& B, const handles::__desc__& a_desc, const handles::__desc__& b_desc)
    {
	    return sub(A, A, B, a_desc, a_desc, b_desc);
    }

    __event__ mul(handles::__memory__& A, const handles::__memory__& B, const handles::__desc__& a_desc, const handles::__desc__& b_desc)
    {
	    return mul(A, A, B, a_desc, a_desc, b_desc);
    }

    __event__ div(handles::__memory__& A, const handles::__memory__& B, const handles::__desc__& a_desc, const handles::__desc__& b_desc)
    {
	    return div(A, A, B, a_desc, a_desc, b_desc);
    }

    __event__ matmul(handles::__memory__& A, const handles::__memory__& B, const handles::__desc__& a_desc, const handles::__desc__& b_desc)
    {
	    return matmul(A, A, B, a_desc, a_desc, b_desc);
    }


    /* RAII handles overloads */

    __event__ add(Memory& C, const Memory& A, const Memory& B, const Descriptor& c_desc, const Descriptor& a_desc, const Descriptor& b_desc)
    {
        handles::__memory__& c_memory = C.handle();
        const handles::__memory__& a_memory = A.handle();
        const handles::__memory__& b_memory = B.handle();

        const handles::__desc__& c_descriptor = c_desc.handle();
        const handles::__desc__& a_descriptor = a_desc.handle();
        const handles::__desc__& b_descriptor = b_desc.handle();

        return add(c_memory, a_memory, b_memory, c_descriptor, a_descriptor, b_descriptor);
    }

    __event__ sub(Memory& C, const Memory& A, const Memory& B, const Descriptor& c_desc, const Descriptor& a_desc, const Descriptor& b_desc)
    {
        handles::__memory__& c_memory = C.handle();
        const handles::__memory__& a_memory = A.handle();
        const handles::__memory__& b_memory = B.handle();

        const handles::__desc__& c_descriptor = c_desc.handle();
        const handles::__desc__& a_descriptor = a_desc.handle();
        const handles::__desc__& b_descriptor = b_desc.handle();

        return sub(c_memory, a_memory, b_memory, c_descriptor, a_descriptor, b_descriptor);
    }

    __event__ mul(Memory& C, const Memory& A, const Memory& B, const Descriptor& c_desc, const Descriptor& a_desc, const Descriptor& b_desc)
    {
        handles::__memory__& c_memory = C.handle();
        const handles::__memory__& a_memory = A.handle();
        const handles::__memory__& b_memory = B.handle();

        const handles::__desc__& c_descriptor = c_desc.handle();
        const handles::__desc__& a_descriptor = a_desc.handle();
        const handles::__desc__& b_descriptor = b_desc.handle();

        return mul(c_memory, a_memory, b_memory, c_descriptor, a_descriptor, b_descriptor);
    }

    __event__ div(Memory& C, const Memory& A, const Memory& B, const Descriptor& c_desc, const Descriptor& a_desc, const Descriptor& b_desc)
    {
        handles::__memory__& c_memory = C.handle();
        const handles::__memory__& a_memory = A.handle();
        const handles::__memory__& b_memory = B.handle();

        const handles::__desc__& c_descriptor = c_desc.handle();
        const handles::__desc__& a_descriptor = a_desc.handle();
        const handles::__desc__& b_descriptor = b_desc.handle();

        return div(c_memory, a_memory, b_memory, c_descriptor, a_descriptor, b_descriptor);
    }

    
    __event__ matmul(Memory& C, const Memory& A, const Memory& B, const Descriptor& c_desc, const Descriptor& a_desc, const Descriptor& b_desc)
    {
        handles::__memory__& c_memory = C.handle();
        const handles::__memory__& a_memory = A.handle();
        const handles::__memory__& b_memory = B.handle();

        const handles::__desc__& c_descriptor = c_desc.handle();
        const handles::__desc__& a_descriptor = a_desc.handle();
        const handles::__desc__& b_descriptor = b_desc.handle();

        return matmul(c_memory, a_memory, b_memory, c_descriptor, a_descriptor, b_descriptor);
    }


    /* Objective tensors overloads */

    __event__ add(Tensor& C, const Tensor& A, const Tensor& B)
    {
        handles::__memory__& c_memory = C.memory()->handle();
        const handles::__memory__& a_memory = A.memory()->handle();
        const handles::__memory__& b_memory = B.memory()->handle();

        const handles::__desc__& c_descriptor = C.descriptor()->handle();
        const handles::__desc__& a_descriptor = A.descriptor()->handle();
        const handles::__desc__& b_descriptor = B.descriptor()->handle();

        return add(c_memory, a_memory, b_memory, c_descriptor, a_descriptor, b_descriptor);    
    }

    __event__ sub(Tensor& C, const Tensor& A, const Tensor& B)
    {
        handles::__memory__& c_memory = C.memory()->handle();
        const handles::__memory__& a_memory = A.memory()->handle();
        const handles::__memory__& b_memory = B.memory()->handle();

        const handles::__desc__& c_descriptor = C.descriptor()->handle();
        const handles::__desc__& a_descriptor = A.descriptor()->handle();
        const handles::__desc__& b_descriptor = B.descriptor()->handle();

        return sub(c_memory, a_memory, b_memory, c_descriptor, a_descriptor, b_descriptor);   
    }

    __event__ mul(Tensor& C, const Tensor& A, const Tensor& B)
    {
        handles::__memory__& c_memory = C.memory()->handle();
        const handles::__memory__& a_memory = A.memory()->handle();
        const handles::__memory__& b_memory = B.memory()->handle();

        const handles::__desc__& c_descriptor = C.descriptor()->handle();
        const handles::__desc__& a_descriptor = A.descriptor()->handle();
        const handles::__desc__& b_descriptor = B.descriptor()->handle();

        return mul(c_memory, a_memory, b_memory, c_descriptor, a_descriptor, b_descriptor);   
    }

    __event__ div(Tensor& C, const Tensor& A, const Tensor& B)
    {
        handles::__memory__& c_memory = C.memory()->handle();
        const handles::__memory__& a_memory = A.memory()->handle();
        const handles::__memory__& b_memory = B.memory()->handle();

        const handles::__desc__& c_descriptor = C.descriptor()->handle();
        const handles::__desc__& a_descriptor = A.descriptor()->handle();
        const handles::__desc__& b_descriptor = B.descriptor()->handle();

        return div(c_memory, a_memory, b_memory, c_descriptor, a_descriptor, b_descriptor);   
    }

    
    __event__ matmul(Tensor& C, const Tensor& A, const Tensor& B)
    {
        handles::__memory__& c_memory = C.memory()->handle();
        const handles::__memory__& a_memory = A.memory()->handle();
        const handles::__memory__& b_memory = B.memory()->handle();

        const handles::__desc__& c_descriptor = C.descriptor()->handle();
        const handles::__desc__& a_descriptor = A.descriptor()->handle();
        const handles::__desc__& b_descriptor = B.descriptor()->handle();

        return matmul(c_memory, a_memory, b_memory, c_descriptor, a_descriptor, b_descriptor);   
    }

}
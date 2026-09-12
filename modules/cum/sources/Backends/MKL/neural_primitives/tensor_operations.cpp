/**----------------------------------------------
 * ?                    ABOUT
 * @author      : Świadomy Mężczyzna
 * @email       : 
 * @file        : modules/cum/sources/Backends/MKL/neural_primitives/tensor_operations.hpp
 * @createdOn   : 09/12/26 
 * @description : ND tensors operation declarations
 *---------------------------------------------**/

#include "cum/detail/vendor/oneapi/opaque_types.hpp"

#include "cum/neural_primitives/tensor_operations.hpp"

namespace cum::neural_primitives
{
    /* Raw handles overloads */
    __event__ add(handles::__memory__ C, handles::__memory__ A, handles::__memory__ B, handles::__desc__ c_desc, handles::__desc__ a_desc, handles::__desc__ b_desc)
    {

    }

    __event__ sum(handles::__memory__ C, handles::__memory__ A, handles::__memory__ B, handles::__desc__ c_desc, handles::__desc__ a_desc, handles::__desc__ b_desc)
    {

    }

    __event__ mul(handles::__memory__ C, handles::__memory__ A, handles::__memory__ B, handles::__desc__ c_desc, handles::__desc__ a_desc, handles::__desc__ b_desc)
    {

    }

    __event__ div(handles::__memory__ C, handles::__memory__ A, handles::__memory__ B, handles::__desc__ c_desc, handles::__desc__ a_desc, handles::__desc__ b_desc)
    {

    }

    
    __event__ matmul(handles::__memory__ C, handles::__memory__ A, handles::__memory__ B, handles::__desc__ c_desc, handles::__desc__ a_desc, handles::__desc__ b_desc)
    {

    }


    /* RAII handles overloads */

    __event__ add(Memory& C, const Memory& A, const Memory& B, const Descriptor& c_desc, const Descriptor& a_desc, const Descriptor& b_desc)
    {

    }

    __event__ sum(Memory& C, const Memory& A, const Memory& B, const Descriptor& c_desc, const Descriptor& a_desc, const Descriptor& b_desc)
    {

    }

    __event__ mul(Memory& C, const Memory& A, const Memory& B, const Descriptor& c_desc, const Descriptor& a_desc, const Descriptor& b_desc)
    {

    }

    __event__ div(Memory& C, const Memory& A, const Memory& B, const Descriptor& c_desc, const Descriptor& a_desc, const Descriptor& b_desc)
    {

    }

    
    __event__ matmul(Memory& C, const Memory& A, const Memory& B, const Descriptor& c_desc, const Descriptor& a_desc, const Descriptor& b_desc)
    {

    }


    /* Objective tensors overloads */

    __event__ add(Tensor& C, const Tensor& A, const Tensor& B)
    {

    }

    __event__ sub(Tensor& C, const Tensor& A, const Tensor& B)
    {

    }

    __event__ mul(Tensor& C, const Tensor& A, const Tensor& B)
    {

    }

    __event__ div(Tensor& C, const Tensor& A, const Tensor& B)
    {

    }

    
    __event__ matmul(Tensor& C, const Tensor& A, const Tensor& B)
    {

    }

}
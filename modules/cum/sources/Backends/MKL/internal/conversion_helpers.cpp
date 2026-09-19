//
// Created by jaro on 9/10/26.
//

#include <oneapi/dnnl/dnnl.hpp>

#include "cum/detail/vendor/oneapi/conversion_helpers.hpp"

namespace cum
{
    dnnl::algorithm dnnl_algorithm(functions::function_id id)
    {
        switch (id)
        {
            case functions::function_id::relu: return dnnl::algorithm::eltwise_relu;
            case functions::function_id::sigmoid: return dnnl::algorithm::eltwise_logistic;
            case functions::function_id::tanh: return dnnl::algorithm::eltwise_tanh;
            case functions::function_id::elu: return dnnl::algorithm::eltwise_elu;
            case functions::function_id::leaky_relu: return dnnl::algorithm::eltwise_relu;
            case functions::function_id::sqrt: return dnnl::algorithm::eltwise_sqrt;
                // case functions::function_id::softmax: return dnnl::algorithm::softmax_accurate;
            case functions::function_id::gelu: return dnnl::algorithm::eltwise_gelu_tanh;
            default: return dnnl::algorithm::undef;
        }
    }

    
    dnnl::memory::data_type dnnl_data_type(datatype dtype)
    {
        switch(dtype)
        {
            case datatype::FP64: return dnnl::memory::data_type::f64;
            case datatype::FP32: return dnnl::memory::data_type::f32;
            case datatype::FP16: return dnnl::memory::data_type::f16;
            case datatype::BF16: return dnnl::memory::data_type::bf16;
            case datatype::FP8:  return dnnl::memory::data_type::f8_e4m3;
            case datatype::S64:   return dnnl::memory::data_type::s64;
            case datatype::S32:   return dnnl::memory::data_type::s32;
            case datatype::S16:   return dnnl::memory::data_type::undef;
            case datatype::S8:   return dnnl::memory::data_type::s8;
            case datatype::U64:   return dnnl::memory::data_type::undef;
            case datatype::U32:   return dnnl::memory::data_type::undef;
            case datatype::U16:   return dnnl::memory::data_type::undef;
            case datatype::U8:   return dnnl::memory::data_type::u8;
            default: return dnnl::memory::data_type::undef;
        }
    }

    datatype cum_data_type(dnnl::memory::data_type dtype)
    {
        switch(dtype)
        {
            case dnnl::memory::data_type::f64:      return datatype::FP64;
            case dnnl::memory::data_type::f32:      return datatype::FP32;
            case dnnl::memory::data_type::f16:      return datatype::FP16;
            case dnnl::memory::data_type::bf16:     return datatype::BF16;
            case dnnl::memory::data_type::f8_e4m3:  return datatype::FP8;
            case dnnl::memory::data_type::s8:       return datatype::S8;
            case dnnl::memory::data_type::u8:       return datatype::U8;
            case dnnl::memory::data_type::s32:  return datatype::S32;
            default:                            return datatype::UNDEF;
        }
    }

    dnnl::memory::format_tag dnnl_format_tag(layout format)
    {
        using tag = dnnl::memory::format_tag;

        switch(format)
        {
            case layout::ANY:          return tag::any;

            case layout::A:            return tag::a;

            case layout::AB:           return tag::ab;
            case layout::BA:           return tag::ba;

            case layout::ABC:          return tag::abc;
            case layout::ACB:          return tag::acb;
            case layout::BAC:          return tag::bac;
            case layout::BCA:          return tag::bca;
            case layout::CBA:          return tag::cba;

            case layout::ABCD:         return tag::abcd;
            case layout::ABDC:         return tag::abdc;
            case layout::ACBD:         return tag::acbd;
            case layout::ACDB:         return tag::acdb;
            case layout::ADBC:         return tag::adbc;
            case layout::BACD:         return tag::bacd;
            case layout::BCDA:         return tag::bcda;
            case layout::CDBA:         return tag::cdba;
            case layout::DCAB:         return tag::dcab;

            case layout::ABCDE:        return tag::abcde;
            case layout::ABDEC:        return tag::abdec;
            case layout::ACBDE:        return tag::acbde;
            case layout::ACDEB:        return tag::acdeb;
            case layout::BACDE:        return tag::bacde;
            case layout::BCDEA:        return tag::bcdea;
            case layout::CDEBA:        return tag::cdeba;
            case layout::DECAB:        return tag::decab;
            case layout::ABCED:        return tag::abced;

            case layout::ABCDEF:       return tag::abcdef;
            case layout::ABDFCE:       return tag::abdfce;
            case layout::ACBDEF:       return tag::acbdef;
            case layout::ABDEFC:       return tag::abdefc;
            case layout::DEFCAB:       return tag::defcab;
            case layout::ABCDFE:       return tag::abcdfe;

            case layout::ABCDEFG:      return tag::abcdefg;
            case layout::ABCDEGF:      return tag::abcdegf;

            case layout::ABCDEFGH:     return tag::abcdefgh;
            case layout::ABCDEFHG:     return tag::abcdefhg;

            case layout::ABCDEFGHI:    return tag::abcdefghi;
            case layout::ABCDEFGIH:    return tag::abcdefgih;

            case layout::ABCDEFGHIJ:   return tag::abcdefghij;
            case layout::ABCDEFGHJI:   return tag::abcdefghji;

            case layout::ABCDEFGHIJK:  return tag::abcdefghijk;
            case layout::ABCDEFGHIKJ:  return tag::abcdefghikj;

            case layout::ABCDEFGHIJKL: return tag::abcdefghijkl;
            case layout::ABCDEFGHIJLK: return tag::abcdefghijlk;

            case layout::UNDEF:        return tag::undef;
        }

        return tag::undef;
    }

    layout cum_format_tag(dnnl::memory::format_tag format)
    {
        using tag = dnnl::memory::format_tag;

        switch(format)
        {
            case tag::any:          return layout::ANY;

            case tag::a:            return layout::A;

            case tag::ab:           return layout::AB;
            case tag::ba:           return layout::BA;

            case tag::abc:          return layout::ABC;
            case tag::acb:          return layout::ACB;
            case tag::bac:          return layout::BAC;
            case tag::bca:          return layout::BCA;
            case tag::cba:          return layout::CBA;

            case tag::abcd:         return layout::ABCD;
            case tag::abdc:         return layout::ABDC;
            case tag::acbd:         return layout::ACBD;
            case tag::acdb:         return layout::ACDB;
            case tag::adbc:         return layout::ADBC;
            case tag::bacd:         return layout::BACD;
            case tag::bcda:         return layout::BCDA;
            case tag::cdba:         return layout::CDBA;
            case tag::dcab:         return layout::DCAB;

            case tag::abcde:        return layout::ABCDE;
            case tag::abdec:        return layout::ABDEC;
            case tag::acbde:        return layout::ACBDE;
            case tag::acdeb:        return layout::ACDEB;
            case tag::bacde:        return layout::BACDE;
            case tag::bcdea:        return layout::BCDEA;
            case tag::cdeba:        return layout::CDEBA;
            case tag::decab:        return layout::DECAB;
            case tag::abced:        return layout::ABCED;

            case tag::abcdef:       return layout::ABCDEF;
            case tag::abdfce:       return layout::ABDFCE;
            case tag::acbdef:       return layout::ACBDEF;
            case tag::abdefc:       return layout::ABDEFC;
            case tag::defcab:       return layout::DEFCAB;
            case tag::abcdfe:       return layout::ABCDFE;

            case tag::abcdefg:      return layout::ABCDEFG;
            case tag::abcdegf:      return layout::ABCDEGF;

            case tag::abcdefgh:     return layout::ABCDEFGH;
            case tag::abcdefhg:     return layout::ABCDEFHG;

            case tag::abcdefghi:    return layout::ABCDEFGHI;
            case tag::abcdefgih:    return layout::ABCDEFGIH;

            case tag::abcdefghij:   return layout::ABCDEFGHIJ;
            case tag::abcdefghji:   return layout::ABCDEFGHJI;

            case tag::abcdefghijk:  return layout::ABCDEFGHIJK;
            case tag::abcdefghikj:  return layout::ABCDEFGHIKJ;

            case tag::abcdefghijkl: return layout::ABCDEFGHIJKL;
            case tag::abcdefghijlk: return layout::ABCDEFGHIJLK;

            case tag::undef:        return layout::UNDEF;

            default:                return layout::UNDEF;
        }
    }
    
}
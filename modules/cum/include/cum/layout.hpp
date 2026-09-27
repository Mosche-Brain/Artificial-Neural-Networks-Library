/*
 * @author: jaro
 * @name:   layou
 * @file:   modules/cum/layou.hpp
 * @date:   19 September 2026 11:00:58
 */

#pragma once

namespace cum
{
    /*
     * @info: This part was derived from one of the oneDNN headers
     * @source: https://github.com/uxlfoundation/oneDNN/blob/main/include/oneapi/dnnl/dnnl.hpp
     * @license: Apache 2.0
     */
    enum class layout : unsigned char
    {
        UNDEF,
        ANY,

        A,

        AB, // Row Major
        BA, // Col Major

        ABC,
        ACB,
        BAC,
        BCA,
        CBA,

        ABCD,
        ABDC,
        ACBD,
        ACDB,
        ADBC,
        BACD,
        BCDA,
        CDBA,
        DCAB,

        ABCDE,
        ABDEC,
        ACBDE,
        ACDEB,
        BACDE,
        BCDEA,
        CDEBA,
        DECAB,
        ABCED,

        ABCDEF,
        ABDFCE,
        ACBDEF,
        ABDEFC,
        DEFCAB,
        ABCDFE,

        ABCDEFG,
        ABCDEGF,

        ABCDEFGH,
        ABCDEFHG,

        ABCDEFGHI,
        ABCDEFGIH,

        ABCDEFGHIJ,
        ABCDEFGHJI,

        ABCDEFGHIJK,
        ABCDEFGHIKJ,

        ABCDEFGHIJKL,
        ABCDEFGHIJLK,

        // 1D
        X = A,

        // Activations
        NC    = AB,
        CN    = BA,
        NCW   = ABC,
        NWC   = ACB,
        NCHW  = ABCD,
        NHWC  = ACDB,
        CHWN  = BCDA,
        NCDHW = ABCDE,
        NDHWC = ACDEB,

        // Weights
        OI    = AB,
        IO    = BA,

        OIW   = ABC,
        OWI   = ACB,
        WIO   = CBA,
        IWO   = BCA,

        OIHW  = ABCD,
        HWIO  = CDBA,
        OHWI  = ACDB,
        IHWO  = BCDA,
        IOHW  = BACD,

        OIDHW = ABCDE,
        DHWIO = CDEBA,
        ODHWI = ACDEB,
        IODHW = BACDE,
        IDHWO = BCDEA,

        // Grouped weights
        GOIW   = ABCD,
        GOWI   = ABDC,
        WIGO   = DCAB,

        GOHWI  = ABDEC,
        GOIHW  = ABCDE,
        HWIGO  = DECAB,
        GIOHW  = ACBDE,

        GOIDHW = ABCDEF,
        GIODHW = ACBDEF,
        GODHWI = ABDEFC,
        DHWIGO = DEFCAB,

        // RNN
        TN    = AB,
        NT    = BA,
        TNC   = ABC,
        NTC   = BAC,

        LDNC  = ABCD,
        LDIGO = ABCDE,
        LDGOI = ABDEC,
        LDIO  = ABCD,
        LDOI  = ABDC,
        LDGO  = ABCD,
    };

}
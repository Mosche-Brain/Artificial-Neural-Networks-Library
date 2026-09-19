/*
 * @author: jaro
 * @name:   layou
 * @file:   modules/cum/layou.hpp
 * @date:   19 September 2026 11:00:58
 */

#pragma once

namespace cum
{

    // enum class layout : std::uint8_t {
    //     ANY,
    //
    //     A,
    //     AB,
    //
    //     BA,
    //
    //     X = A, // Vector
    //
    //     NC = AB,
    //     BC = NC,       // Batch, Channels / Features
    //
    //     CN = BA,
    //
    //     OI = AB,
    //     IO = BA,
    //
    //     NCHW,
    //     BCHW = NCHW,   // Batch, Channels, Height, Width
    //
    //     NHWC,
    //     BHWC = NHWC,   // Batch, Height, Width, Channels
    //
    //     OIHW,
    //
    //     HWIO,
    //
    //     NCDHW,
    //     BCDHW = NCDHW, // Batch, Channels, Depth, Height, Width
    //
    //     NDHWC,
    //     BDHWC = NDHWC, // Batch, Depth, Height, Width, Channels
    //
    //     TNC,
    //     SBC = TNC,     // Sequence, Batch, Channels
    //     SBE = TNC,     // Sequence, Batch, Embedding
    //
    //     NTC,
    //     BSC = NTC,     // Batch, Sequence, Channels
    //     BSE = NTC,     // Batch, Sequence, Embedding
    //
    //     STRIDED,
    //     UNDEF
    // };

    /*
     * @info: This part was derived from one of the oneDNN header
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
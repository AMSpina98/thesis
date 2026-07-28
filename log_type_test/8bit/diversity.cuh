#pragma once


#include "my_type.cuh"


__device__ NL dMul(NL A, NL B) {
    uint8_t mantissaA = (A & NLFormat::MAX_MANTISSA);
    uint8_t mantissaB = (B & NLFormat::MAX_MANTISSA);
    uint8_t signA = (A >> 7) & 0x1;
    uint8_t signB = (B >> 7) & 0x1;
    float xa = (__logf((float)mantissaA) - NLFormat::MANTISSA_FACTOR) * NLFormat::INV_LOG_BASE;
    float xb = (__logf((float)mantissaB) - NLFormat::MANTISSA_FACTOR) * NLFormat::INV_LOG_BASE;
    float xc = xa * xb;
    float I = __expf(- xc * NLFormat::LOG_BASE) * NLFormat::MANTISSA_SCALE;
    uint8_t mantissaC = __float2uint_rn(I);
    if(mantissaC > NLFormat::MAX_MANTISSA) mantissaC = NLFormat::MAX_MANTISSA;
    return ((signA ^ signB) << 7) | mantissaC;
}


__device__ NL dDiv(NL A, NL B) {
    uint8_t mantissaA = (A & NLFormat::MAX_MANTISSA);
    uint8_t mantissaB = (B & NLFormat::MAX_MANTISSA);
    uint8_t signA = (A >> 7) & 0x1;
    uint8_t signB = (B >> 7) & 0x1;
    float xa = (__logf((float)mantissaA) - NLFormat::MANTISSA_FACTOR) * NLFormat::INV_LOG_BASE;
    float xb = (__logf((float)mantissaB) - NLFormat::MANTISSA_FACTOR) * NLFormat::INV_LOG_BASE;
    float xc = xa / xb;
    float I = __expf(- xc * NLFormat::LOG_BASE) * NLFormat::MANTISSA_SCALE;
    uint8_t mantissaC = __float2uint_rn(I);
    if(mantissaC > NLFormat::MAX_MANTISSA) mantissaC = NLFormat::MAX_MANTISSA;
    return ((signA ^ signB) << 7) | mantissaC;
}

__device__ NL dAdd(NL A, NL B) {
    uint8_t mantissaA = (A & NLFormat::MAX_MANTISSA);
    uint8_t mantissaB = (B & NLFormat::MAX_MANTISSA);
    uint8_t signA = (A >> 7) & 0x1;
    uint8_t signB = (B >> 7) & 0x1;
    uint8_t mantissaTempA = max(mantissaA, mantissaB);
    uint8_t mantissaTempB;
    if(mantissaA == mantissaTempA){
        mantissaTempB = mantissaB;
    } else{
        mantissaTempB = mantissaA;
        uint8_t temp = signA;
        signA = signB;
        signB = temp;
    }
    if(signA == signB){
        uint64_t mantissa = ((uint64_t)mantissaTempA * (uint64_t)mantissaTempB) >> NLFormat::MANTISSA_LENGTH;
        if(mantissa>MAX_MANTISSA) mantissa=MAX_MANTISSA;
        return (signA << 7) | mantissa;
    } else{
        uint64_t mantissa = (((uint64_t)mantissaTempB << NLFormat::MANTISSA_LENGTH) / (uint64_t)mantissaTempA);
        return (signB << 7) | mantissa;
    }
}


__device__ NL dSub(NL A, NL B){
    return dAdd(A, (B ^ (1u << 7)));
}

#pragma once


#include "my_type.cuh"


__device__ NL dMul(NL A, NL B) {
    uint32_t baseA = (A >> NLFormat::MANTISSA_LENGTH) & 0xF;
    uint32_t baseB = (B >> NLFormat::MANTISSA_LENGTH) & 0xF;
    uint32_t baseC = max(baseA, baseB);
    uint32_t mantissaA = (A & NLFormat::MAX_MANTISSA);
    uint32_t mantissaB = (B & NLFormat::MAX_MANTISSA);
    uint32_t signA = (A >> 31) & 0x1;
    uint32_t signB = (B >> 31) & 0x1;
    double exp = (__logf(mantissaA) - NLFormat::MANTISSA_FACTOR) * NLFormat::inv_log_base[baseA] * (__logf(mantissaB) - NLFormat::MANTISSA_FACTOR) * NLFormat::inv_log_base[baseB];
    double I = __expf(- exp * NLFormat::log_base[baseC]) * NLFormat::MANTISSA_SCALE;
    uint32_t mantissaC = (uint32_t)I;
    if(mantissaC > NLFormat::MAX_MANTISSA) mantissaC = NLFormat::MAX_MANTISSA;
    return ((signA ^ signB) << 31) | (baseC << NLFormat::MANTISSA_LENGTH) | mantissaC;
}


__device__ NL dDiv(NL A, NL B) {
    uint32_t baseA = (A >> NLFormat::MANTISSA_LENGTH) & 0xF;
    uint32_t baseB = (B >> NLFormat::MANTISSA_LENGTH) & 0xF;
    uint32_t baseC = max(baseA, baseB);
    uint32_t mantissaA = (A & NLFormat::MAX_MANTISSA);
    uint32_t mantissaB = (B & NLFormat::MAX_MANTISSA);
    uint32_t signA = (A >> 31) & 0x1;
    uint32_t signB = (B >> 31) & 0x1;
    double exp = ((__logf(mantissaA) - NLFormat::MANTISSA_FACTOR) * NLFormat::inv_log_base[baseA]) / ((__logf(mantissaB) - NLFormat::MANTISSA_FACTOR) * NLFormat::inv_log_base[baseB]);
    double I = __expf(- exp * NLFormat::log_base[baseC]) * NLFormat::MANTISSA_SCALE;
    uint32_t mantissaC = (uint32_t)I;
    if(mantissaC > NLFormat::MAX_MANTISSA) mantissaC = NLFormat::MAX_MANTISSA;
    return ((signA ^ signB) << 31) | (baseC << NLFormat::MANTISSA_LENGTH) | mantissaC;
}

__device__ NL dConvertBase(NL A, uint32_t newBase) {
    uint32_t baseA = (A >> NLFormat::MANTISSA_LENGTH) & 0xF;
    if(baseA == newBase){
        return A;
    }
    uint64_t mantissaA = (A & NLFormat::MAX_MANTISSA) * NLFormat::base_changers[newBase][baseA];
    uint32_t signA = (A >> 31) & 0x1;
    return (signA << 31) | (newBase << NLFormat::MANTISSA_LENGTH) | (uint32_t)mantissaA;
}

__device__ NL dAdd(NL A, NL B) {
    uint32_t baseA = (A >> NLFormat::MANTISSA_LENGTH) & 0xF;
    uint32_t baseB = (B >> NLFormat::MANTISSA_LENGTH) & 0xF;
    if(baseA != baseB){
        //  convert A and B to the same base
        if(baseA > baseB){
            B = dConvertBase(B, baseA);
        } else{
            A = dConvertBase(A, baseB);
            baseA = baseB;
        }
    }
    uint32_t mantissaA = A & NLFormat::MAX_MANTISSA;
    uint32_t mantissaB = B & NLFormat::MAX_MANTISSA;
    uint32_t signA = (A >> 31) & 0x1;
    uint32_t signB = (B >> 31) & 0x1;
    uint32_t mantissaTempA = max(mantissaA, mantissaB);
    uint32_t mantissaTempB;
    if(mantissaA == mantissaTempA){
        mantissaTempB = mantissaB;
    } else{
        mantissaTempB = mantissaA;
        uint32_t temp = signA;
        signA = signB;
        signB = temp;
    }
    if(signA == signB){
        uint64_t mantissa = ((uint64_t)mantissaTempA * (uint64_t)mantissaTempB) >> NLFormat::MANTISSA_LENGTH;
        if(mantissa>MAX_MANTISSA) mantissa=MAX_MANTISSA;
        return (signA << 31) | (baseA << NLFormat::MANTISSA_LENGTH) | mantissa;
    } else{
        uint64_t mantissa = (((uint64_t)mantissaTempB << NLFormat::MANTISSA_LENGTH) / (uint64_t)mantissaTempA);
        return (signB << 31) | (baseA << NLFormat::MANTISSA_LENGTH) | mantissa;
    }
}


__device__ NL dSub(NL A, NL B){
    return dAdd(A, (B ^ (1u << 31)));
}

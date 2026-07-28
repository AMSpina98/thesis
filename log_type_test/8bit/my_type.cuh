#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#ifdef __CUDA_BF16_H__
#include <cuda_bf16.h>
#endif
#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#define HAS_FP8 1
#else
#define HAS_FP8 0
#endif

namespace NLFormat {
    constexpr int SIGN_BITS = 1;
    constexpr int MANTISSA_LENGTH = 7;
    constexpr double BASE = 2.0;;
    constexpr double LN2 = 0.6931471805599453;
    constexpr double MANTISSA_FACTOR = static_cast<double>(MANTISSA_LENGTH) * LN2;
    constexpr uint32_t MANTISSA_SCALE = 1u << MANTISSA_LENGTH;
    constexpr uint32_t MAX_MANTISSA = (1u << MANTISSA_LENGTH) - 1;
    constexpr float LOG_BASE = 0.6931471805599453f;
    constexpr float INV_LOG_BASE = 1.4426950408889634f;
}


//Converter
template<typename T>
struct Converter{
	__device__ static NL to(T x);
	__device__ static T from(NL x);
};


template<>
struct Converter<int> {
    __device__ static NL to(int x){
        uint32_t sign = (x < 0);
        float ax = fabs(x);
        float I = __expf(-ax * NLFormat::LOG_BASE);
        uint32_t mantissa = __float2uint_rn(I * NLFormat::MANTISSA_SCALE);
        if (mantissa > NLFormat::MAX_MANTISSA) mantissa = NLFormat::MAX_MANTISSA;
        return (sign << 7) |  mantissa;
    }

    __device__ static int from(NL x){
        uint32_t sign = (x >> 7) & 0x1;
        int32_t mantissa = x & ((1u << NLFormat::MANTISSA_LENGTH) - 1);
        float value = (NLFormat::MANTISSA_FACTOR - __logf((float)mantissa)) * NLFormat::INV_LOG_BASE;
        if (sign) value = -value;
        return __float2int_rn(value);
    }
};


template<>
struct Converter<unsigned int> {
    __device__ static NL to(unsigned int x){
        float ax = (float)x;
        float I = __expf(-ax * NLFormat::LOG_BASE);
        uint32_t mantissa = __float2uint_rn(I * NLFormat::MANTISSA_SCALE);
        mantissa = max(1u, mantissa);
        mantissa = min(NLFormat::MAX_MANTISSA, mantissa);
        return (0u << 7) | mantissa;
    }


    __device__ static unsigned int from(NL x){
        uint32_t mantissa = x & NLFormat::MAX_MANTISSA;
        float value = (NLFormat::MANTISSA_FACTOR - __logf((float)mantissa)) * NLFormat::INV_LOG_BASE;
        return __float2uint_rn(value);
    }
};

template<>
struct Converter<double> {
    __device__ static NL to(double x){
        uint32_t sign = (x < 0.0);
        double ax = fabs(x);
        double I = __expf(-ax * NLFormat::LOG_BASE);
        uint32_t mantissa = __double2uint_rn(I * NLFormat::MANTISSA_SCALE);
        if (mantissa > NLFormat::MAX_MANTISSA) mantissa = NLFormat::MAX_MANTISSA;
        return (sign << 7) | mantissa;
    }

    __device__ static double from(NL x) {
        uint32_t sign = (x >> 7) & 0x1;
        int32_t mantissa = x & ((1u << NLFormat::MANTISSA_LENGTH) - 1);
        double value = (NLFormat::MANTISSA_FACTOR - __logf((float)mantissa)) * NLFormat::INV_LOG_BASE;
        if (sign) value = -value;
        return value;
    }
};

template<>
struct Converter<float> {
	__device__ static NL to(float x){
        uint32_t sign = (x < 0.0);
        float ax = fabs(x);
        float I = __expf(-ax * NLFormat::LOG_BASE);
        uint32_t mantissa = __float2uint_rn(I * NLFormat::MANTISSA_SCALE);
        if (mantissa > NLFormat::MAX_MANTISSA) mantissa = NLFormat::MAX_MANTISSA;
        return (sign << 7) | mantissa;
	}
	__device__ static float from(NL x){
        uint32_t sign = (x >> 7) & 0x1;
        int32_t mantissa = x & ((1u << NLFormat::MANTISSA_LENGTH) - 1);
        float value = (NLFormat::MANTISSA_FACTOR - __logf((float)mantissa)) * NLFormat::INV_LOG_BASE;
        if (sign) value = -value;
        return value;
	}
};
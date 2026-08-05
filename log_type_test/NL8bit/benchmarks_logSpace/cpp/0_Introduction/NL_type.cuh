#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <limits>
#ifdef __CUDA_BF16_H__
#include <cuda_bf16.h>
#endif
#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#define HAS_FP8 1
#else
#define HAS_FP8 0
#endif

struct NL;

namespace NLFormat {

    constexpr int SIGN_BITS = 1;
    constexpr int MANTISSA_LENGTH = 7;
    constexpr uint32_t MAX_MANTISSA = (1u << MANTISSA_LENGTH) - 1;
    #define LOG_ZERO 127
    #define LOG_NAN  255
}

namespace NLMath {

    __host__ __device__ inline float Log(float x) {
    #ifdef __CUDA_ARCH__
        return __logf(x);
    #else
        return logf(x);
    #endif
    }

    __host__ __device__ inline float Exp(float x) {
    #ifdef __CUDA_ARCH__
        return __expf(x);
    #else
        return expf(x);
    #endif
    }

    __host__ __device__ inline float Abs(float x) {
    #ifdef __CUDA_ARCH__
        return fabsf(x);
    #else
        return std::fabs(x);
    #endif
    }

    __host__ __device__ inline double DAbs(double x) {
    #ifdef __CUDA_ARCH__
        return fabs(x);
    #else
        return std::fabs(x);
    #endif
    }

    __host__ __device__ inline int NearbyInt(float x) {
    #ifdef __CUDA_ARCH__
        return __float2int_rn(x);
    #else
        return std::nearbyintf(x);
    #endif
    }

    __host__ __device__ inline float Max(float a, float b) {
    #ifdef __CUDA_ARCH__   
        return fmaxf(a, b);
    #else
        return std::fmax(a, b);
    #endif
    }

    __host__ __device__ inline float Min(float a, float b) {
    #ifdef __CUDA_ARCH__
        return fminf(a, b);
    #else
        return std::fmin(a, b);
    #endif
    }

    __host__ __device__ inline unsigned int NearbyUInt(float x) {
    #ifdef __CUDA_ARCH__
        return __float2uint_rn(x);
    #else
        return static_cast<unsigned int>(std::nearbyintf(x));
    #endif
    }

    __host__ __device__ inline bool IsNaN(float x) {
    #ifdef __CUDA_ARCH__
        return isnan(x);
    #else
        return std::isnan(x);
    #endif
    }

    __host__ __device__ inline bool IsNaNDouble(double x) {
    #ifdef __CUDA_ARCH__
        return isnan(x);
    #else
        return std::isnan(x);
    #endif
    }

    __host__ __device__ inline float Quiet_NAN() {
    #ifdef __CUDA_ARCH__
        return __int_as_float(0x7fc00000);
    #else
        return std::numeric_limits<float>::quiet_NaN();
    #endif
    }

    __host__ __device__ inline double Quiet_NANDouble() {
    #ifdef __CUDA_ARCH__
        return __longlong_as_double(0x7ff8000000000000LL);
    #else
        return std::numeric_limits<double>::quiet_NaN();
    #endif
    }

}


struct NL {private:
    uint8_t value;

    public:
    __host__ __device__
    static NL raw(uint8_t v) {
        return NL(v, RawTag{});
    }

    private:
    struct RawTag {};

    __host__ __device__
    NL(uint8_t v, RawTag) : value(v) {}
    
    public:
    __host__ __device__ NL(float x){   
        if(x == 0.0){
            value = LOG_ZERO;
            return;
        }
        if(NLMath::IsNaN(x)){
            value = LOG_NAN;
            return;
        }
        uint8_t sign = (x < 0.0) ? 1 : 0;
        float y = - NLMath::Log(NLMath::Abs(x));
        uint8_t mantissa = NLMath::NearbyUInt(y * 8.0);
        if(mantissa > NLFormat::MAX_MANTISSA) mantissa = NLFormat::MAX_MANTISSA;
        value = (sign << 7) | mantissa;
    }
    __host__ __device__ NL(double x){  
        if(x == 0.0){
            value = LOG_ZERO;
            return;
        }
        if(NLMath::IsNaNDouble(x)){
            value = LOG_NAN;
            return;
        }
        uint8_t sign = (x < 0.0) ? 1 : 0;
        double y = - NLMath::Log(NLMath::Abs(x));
        uint8_t mantissa = NLMath::NearbyUInt(y * 8.0);
        if(mantissa > NLFormat::MAX_MANTISSA) mantissa = NLFormat::MAX_MANTISSA;
        value = (sign << 7) | mantissa;
    }
    NL() = default;
    __host__ __device__ operator uint8_t() const { return value; }
    

    __host__ __device__ NL operator+(const NL& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return NL::raw(static_cast<uint8_t>(LOG_NAN));
        if(value == LOG_ZERO) return other;
        if(other.value == LOG_ZERO) return *this;
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        float xa = ((float)mantissaA)/8.0f;
        float xb = ((float)mantissaB)/8.0f;
        float xc;
        uint8_t signC;
        if(signA == signB){
            xc = -NLMath::Log(NLMath::Exp(-xa) + NLMath::Exp(-xb));
            signC = signA;
        }else{
            if(xa == xb) return NL::raw(static_cast<uint8_t>(LOG_ZERO));
            xc = -NLMath::Log(NLMath::Abs(NLMath::Exp(-xa) - NLMath::Exp(-xb)));
            if(xa > xb) signC = signA;
            else signC = signB;
        }
        if(xc < 0.0f) xc = 0.0f;
        uint8_t mantissaC = NLMath::NearbyUInt(xc * 8.0f);
        if(mantissaC > NLFormat::MAX_MANTISSA) return NL::raw(static_cast<uint8_t>(LOG_ZERO));
        return NL::raw(static_cast<uint8_t>((signC << 7) | mantissaC));
    }

    __host__ __device__ NL operator+=(const NL& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return NL::raw(static_cast<uint8_t>(LOG_NAN));
        if(value == LOG_ZERO) return other;
        if(other.value == LOG_ZERO) return *this;
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        float xa = ((float)mantissaA)/8.0f;
        float xb = ((float)mantissaB)/8.0f;
        float xc;
        uint8_t signC;
        if(signA == signB){
            xc = -NLMath::Log(NLMath::Exp(-xa) + NLMath::Exp(-xb));
            signC = signA;
        }else{
            if(xa == xb) return NL::raw(static_cast<uint8_t>(LOG_ZERO));
            xc = -NLMath::Log(NLMath::Abs(NLMath::Exp(-xa) - NLMath::Exp(-xb)));
            if(xa > xb) signC = signA;
            else signC = signB;
        }
        if(xc < 0.0f) xc = 0.0f;
        uint8_t mantissaC = NLMath::NearbyUInt(xc * 8.0f);
        if(mantissaC > NLFormat::MAX_MANTISSA) return NL::raw(static_cast<uint8_t>(LOG_ZERO));
        return NL::raw(static_cast<uint8_t>((signC << 7) | mantissaC));
    }

    __host__ __device__ NL operator-(const NL& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return NL::raw(static_cast<uint8_t>(LOG_NAN));
        if(value == LOG_ZERO) return other;
        if(other.value == LOG_ZERO) return *this;
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        float xa = ((float)mantissaA)/8.0f;
        float xb = ((float)mantissaB)/8.0f;
        float xc;
        uint8_t signC;
        if(signA == signB){
            xc = -NLMath::Log(NLMath::Exp(-xa) - NLMath::Exp(-xb));
            signC = signA;
        }else{
            if(xa == xb) return NL::raw(static_cast<uint8_t>(LOG_ZERO));
            xc = -NLMath::Log(NLMath::Abs(NLMath::Exp(-xa) + NLMath::Exp(-xb)));
            if(xa > xb) signC = signA;
            else signC = signB;
        }
        if(xc < 0.0f) xc = 0.0f;
        uint8_t mantissaC = NLMath::NearbyUInt(xc * 8.0f);
        if(mantissaC > NLFormat::MAX_MANTISSA) return NL::raw(static_cast<uint8_t>(LOG_ZERO));
        return NL::raw(static_cast<uint8_t>((signC << 7) | mantissaC));
    }

    __host__ __device__ NL operator-=(const NL& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return NL::raw(static_cast<uint8_t>(LOG_NAN));
        if(value == LOG_ZERO) return other;
        if(other.value == LOG_ZERO) return *this;
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        float xa = ((float)mantissaA)/8.0f;
        float xb = ((float)mantissaB)/8.0f;
        float xc;
        uint8_t signC;
        if(signA == signB){
            xc = -NLMath::Log(NLMath::Exp(-xa) - NLMath::Exp(-xb));
            signC = signA;
        }else{
            if(xa == xb) return NL::raw(static_cast<uint8_t>(LOG_ZERO));
            xc = -NLMath::Log(NLMath::Abs(NLMath::Exp(-xa) + NLMath::Exp(-xb)));
            if(xa > xb) signC = signA;
            else signC = signB;
        }
        if(xc < 0.0f) xc = 0.0f;
        uint8_t mantissaC = NLMath::NearbyUInt(xc * 8.0f);
        if(mantissaC > NLFormat::MAX_MANTISSA) return NL::raw(static_cast<uint8_t>(LOG_ZERO));
        return NL::raw(static_cast<uint8_t>((signC << 7) | mantissaC));
    }

    __host__ __device__ NL operator*(const NL& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return NL::raw(static_cast<uint8_t>(LOG_NAN));
        if(value == LOG_ZERO || other.value == LOG_ZERO) return NL::raw(static_cast<uint8_t>(LOG_ZERO));
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        uint8_t mantissaC = mantissaA + mantissaB;
        if(mantissaC > NLFormat::MAX_MANTISSA) mantissaC = NLFormat::MAX_MANTISSA;
        return NL::raw(static_cast<uint8_t>(((signA ^ signB) << 7) | mantissaC));
    }

    __host__ __device__ NL operator*=(const NL& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return NL::raw(static_cast<uint8_t>(LOG_NAN));
        if(value == LOG_ZERO || other.value == LOG_ZERO) return NL::raw(static_cast<uint8_t>(LOG_ZERO));
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        uint8_t mantissaC = mantissaA + mantissaB;
        if(mantissaC > NLFormat::MAX_MANTISSA) mantissaC = NLFormat::MAX_MANTISSA;
        return NL::raw(static_cast<uint8_t>(((signA ^ signB) << 7) | mantissaC));
    }

    __host__ __device__ NL operator/(const NL& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return NL::raw(static_cast<uint8_t>(LOG_NAN));
        if(value == LOG_ZERO) return NL::raw(static_cast<uint8_t>(LOG_ZERO));
        if(other.value == LOG_ZERO) return NL::raw(static_cast<uint8_t>(LOG_NAN));
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        uint8_t mantissaC = mantissaA > mantissaB ? mantissaA - mantissaB : 0;
        if(mantissaC > NLFormat::MAX_MANTISSA) mantissaC = NLFormat::MAX_MANTISSA;
        return NL::raw(static_cast<uint8_t>(((signA ^ signB) << 7) | mantissaC));
    }

    __host__ __device__ NL operator/=(const NL& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return NL::raw(static_cast<uint8_t>(LOG_NAN));
        if(value == LOG_ZERO) return NL::raw(static_cast<uint8_t>(LOG_ZERO));
        if(other.value == LOG_ZERO) return NL::raw(static_cast<uint8_t>(LOG_NAN));
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        uint8_t mantissaC = mantissaA > mantissaB ? mantissaA - mantissaB : 0;
        if(mantissaC > NLFormat::MAX_MANTISSA) mantissaC = NLFormat::MAX_MANTISSA;
        return NL::raw(static_cast<uint8_t>(((signA ^ signB) << 7) | mantissaC));
    }


    __host__ __device__
    float toFloat() const {
        if(value == LOG_ZERO) return 0.0f;
        if(value == LOG_NAN) return NLMath::Quiet_NAN();
	    uint8_t mantissa = (value & NLFormat::MAX_MANTISSA);
	    uint8_t sign = (value >> 7) & 0x1;
	    float y = ((float)mantissa) / 8.0;
	    float result = NLMath::Exp(-y);
	    return (sign == 1) ? -result : result;   
    }

    __host__ __device__
    double toDouble() const {
        if(value == LOG_ZERO) return 0.0;
        if(value == LOG_NAN) return NLMath::Quiet_NAN();
        uint8_t mantissa = (value & NLFormat::MAX_MANTISSA);
        uint8_t sign = (value >> 7) & 0x1;
        double y = ((double)mantissa) / 8.0;
        double result = NLMath::Exp(-y);
        return (sign == 1) ? -result : result;
    }

};
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

struct NE;

namespace NEFormat {
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

namespace NEMath {

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


struct NE {
    private:
    uint8_t value;

    public:
    __host__ __device__
    static NE raw(uint8_t v) {
        return NE(v, RawTag{});
    }

    private:
    struct RawTag {};

    __host__ __device__
    NE(uint8_t v, RawTag) : value(v) {}
    
    public:
    __host__ __device__ NE(unsigned int x){
        float ax = (float)x;
        float I = NEMath::Exp(-ax * NEFormat::LOG_BASE);
        uint32_t mantissa = NEMath::NearbyUInt(I * NEFormat::MANTISSA_SCALE);
        mantissa = NEMath::Max(1u, mantissa);
        mantissa = NEMath::Min(NEFormat::MAX_MANTISSA, mantissa);
        value = (0u << 7) | mantissa;
    }
    __host__ __device__ NE(float x){
        uint32_t sign = (x < 0.0);
        float ax = NEMath::Abs(x);
        float I = NEMath::Exp(-ax * NEFormat::LOG_BASE);
        uint32_t mantissa = NEMath::NearbyUInt(I * NEFormat::MANTISSA_SCALE);
        if (mantissa > NEFormat::MAX_MANTISSA) mantissa = NEFormat::MAX_MANTISSA;
        value = (sign << 7) | mantissa;
    }
    __host__ __device__ NE(double x){
        uint32_t sign = (x < 0.0);
        double ax = NEMath::DAbs(x);
        double I = NEMath::Exp(-ax * NEFormat::LOG_BASE);
        uint32_t mantissa = NEMath::NearbyUInt(I * NEFormat::MANTISSA_SCALE);
        if (mantissa > NEFormat::MAX_MANTISSA) mantissa = NEFormat::MAX_MANTISSA;
        value = (sign << 7) | mantissa;
    }
    __host__ __device__ NE(int x){
        uint32_t sign = (x < 0);
        float ax = NEMath::Abs(x);
        float I = NEMath::Exp(-ax * NEFormat::LOG_BASE);
        uint32_t mantissa = NEMath::NearbyUInt(I * NEFormat::MANTISSA_SCALE);
        if (mantissa > NEFormat::MAX_MANTISSA) mantissa = NEFormat::MAX_MANTISSA;
        value = (sign << 7) | mantissa;
    }
    NE() = default;


    __host__ __device__ operator uint8_t() const { return value; }

    __host__ __device__ NE operator+(const NE& other) const {
        uint8_t mantissaA = (value & NEFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NEFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        uint8_t mantissaTempA = NEMath::Max(mantissaA, mantissaB);
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
            uint64_t mantissa = ((uint64_t)mantissaTempA * (uint64_t)mantissaTempB) >> NEFormat::MANTISSA_LENGTH;
            if(mantissa>NEFormat::MAX_MANTISSA) mantissa=NEFormat::MAX_MANTISSA;
            return NE::raw(static_cast<uint8_t>((signA << 7) | mantissa));
        } else{
            uint64_t mantissa = (((uint64_t)mantissaTempB << NEFormat::MANTISSA_LENGTH) / (uint64_t)mantissaTempA);
            return NE::raw(static_cast<uint8_t>((signB << 7) | mantissa));
        }
    }

    __host__ __device__ NE operator+=(const NE& other) const {
        uint8_t mantissaA = (value & NEFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NEFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        uint8_t mantissaTempA = NEMath::Max(mantissaA, mantissaB);
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
            uint64_t mantissa = ((uint64_t)mantissaTempA * (uint64_t)mantissaTempB) >> NEFormat::MANTISSA_LENGTH;
            if(mantissa>NEFormat::MAX_MANTISSA) mantissa=NEFormat::MAX_MANTISSA;
            return NE::raw(static_cast<uint8_t>((signA << 7) | mantissa));
        } else{
            uint64_t mantissa = (((uint64_t)mantissaTempB << NEFormat::MANTISSA_LENGTH) / (uint64_t)mantissaTempA);
            return NE::raw(static_cast<uint8_t>((signB << 7) | mantissa));
        }
    }

    __host__ __device__ NE operator-(const NE& other) const {
        uint8_t mantissaA = (value & NEFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NEFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = ((other.value >> 7) & 0x1) ^ 0x1;
        uint8_t mantissaTempA = NEMath::Max(mantissaA, mantissaB);
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
            uint64_t mantissa = ((uint64_t)mantissaTempA * (uint64_t)mantissaTempB) >> NEFormat::MANTISSA_LENGTH;
            if(mantissa>NEFormat::MAX_MANTISSA) mantissa=NEFormat::MAX_MANTISSA;
            return NE::raw(static_cast<uint8_t>((signA << 7) | mantissa));
        } else{
            uint64_t mantissa = (((uint64_t)mantissaTempB << NEFormat::MANTISSA_LENGTH) / (uint64_t)mantissaTempA);
            return NE::raw(static_cast<uint8_t>((signB << 7) | mantissa));
        }
    }

    __host__ __device__ NE operator-=(const NE& other) const {
        uint8_t mantissaA = (value & NEFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NEFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = ((other.value >> 7) & 0x1) ^ 0x1;
        uint8_t mantissaTempA = NEMath::Max(mantissaA, mantissaB);
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
            uint64_t mantissa = ((uint64_t)mantissaTempA * (uint64_t)mantissaTempB) >> NEFormat::MANTISSA_LENGTH;
            if(mantissa>NEFormat::MAX_MANTISSA) mantissa=NEFormat::MAX_MANTISSA;
            return NE::raw(static_cast<uint8_t>((signA << 7) | mantissa));
        } else{
            uint64_t mantissa = (((uint64_t)mantissaTempB << NEFormat::MANTISSA_LENGTH) / (uint64_t)mantissaTempA);
            return NE::raw(static_cast<uint8_t>((signB << 7) | mantissa));
        }
    }

    __host__ __device__ NE operator*(const NE& other) const {
        uint8_t mantissaA = (value & NEFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NEFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        float xa = (NEMath::Log((float)mantissaA) - NEFormat::MANTISSA_FACTOR) * NEFormat::INV_LOG_BASE;
        float xb = (NEMath::Log((float)mantissaB) - NEFormat::MANTISSA_FACTOR) * NEFormat::INV_LOG_BASE;
        float xc = xa * xb;
        float I = NEMath::Exp(- xc * NEFormat::LOG_BASE) * NEFormat::MANTISSA_SCALE;
        uint8_t mantissaC = NEMath::NearbyUInt(I);
        if(mantissaC > NEFormat::MAX_MANTISSA) mantissaC = NEFormat::MAX_MANTISSA;
        return NE::raw(static_cast<uint8_t>((signA ^ signB) << 7) | mantissaC);
    }

    __host__ __device__ NE operator*=(const NE& other) const {
        uint8_t mantissaA = (value & NEFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NEFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        float xa = (NEMath::Log((float)mantissaA) - NEFormat::MANTISSA_FACTOR) * NEFormat::INV_LOG_BASE;
        float xb = (NEMath::Log((float)mantissaB) - NEFormat::MANTISSA_FACTOR) * NEFormat::INV_LOG_BASE;
        float xc = xa * xb;
        float I = NEMath::Exp(- xc * NEFormat::LOG_BASE) * NEFormat::MANTISSA_SCALE;
        uint8_t mantissaC = NEMath::NearbyUInt(I);
        if(mantissaC > NEFormat::MAX_MANTISSA) mantissaC = NEFormat::MAX_MANTISSA;
        return NE::raw(static_cast<uint8_t>((((signA ^ signB) << 7) | mantissaC)));
    }

    __host__ __device__ NE operator/(const NE& other) const {
        uint8_t mantissaA = (value & NEFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NEFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        float xa = (NEMath::Log((float)mantissaA) - NEFormat::MANTISSA_FACTOR) * NEFormat::INV_LOG_BASE;
        float xb = (NEMath::Log((float)mantissaB) - NEFormat::MANTISSA_FACTOR) * NEFormat::INV_LOG_BASE;
        float xc = xa / xb;
        float I = NEMath::Exp(- xc * NEFormat::LOG_BASE) * NEFormat::MANTISSA_SCALE;
        uint8_t mantissaC = NEMath::NearbyUInt(I);
        if(mantissaC > NEFormat::MAX_MANTISSA) mantissaC = NEFormat::MAX_MANTISSA;
        return NE::raw(static_cast<uint8_t>((((signA ^ signB) << 7) | mantissaC)));
    }

    __host__ __device__ NE operator/=(const NE& other) const {
        uint8_t mantissaA = (value & NEFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NEFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        float xa = (NEMath::Log((float)mantissaA) - NEFormat::MANTISSA_FACTOR) * NEFormat::INV_LOG_BASE;
        float xb = (NEMath::Log((float)mantissaB) - NEFormat::MANTISSA_FACTOR) * NEFormat::INV_LOG_BASE;
        float xc = xa / xb;
        float I = NEMath::Exp(- xc * NEFormat::LOG_BASE) * NEFormat::MANTISSA_SCALE;
        uint8_t mantissaC = NEMath::NearbyUInt(I);
        if(mantissaC > NEFormat::MAX_MANTISSA) mantissaC = NEFormat::MAX_MANTISSA;
        return NE::raw(static_cast<uint8_t>((((signA ^ signB) << 7) | mantissaC)));
    }

    __host__ __device__
    float toFloat() const {
        uint32_t sign = (value >> 7) & 0x1;
        int32_t mantissa = value & ((1u << NEFormat::MANTISSA_LENGTH) - 1);
        float value = (NEFormat::MANTISSA_FACTOR - NEMath::Log((float)mantissa)) * NEFormat::INV_LOG_BASE;
        if (sign) value = -value;
        return value; 
    }

    __host__ __device__
    int toInt() const {
        uint32_t sign = (value >> 7) & 0x1;
        int32_t mantissa = value & ((1u << NEFormat::MANTISSA_LENGTH) - 1);
        float value = (NEFormat::MANTISSA_FACTOR - NEMath::Log((float)mantissa)) * NEFormat::INV_LOG_BASE;
        if (sign) value = -value;
        return NEMath::NearbyInt(value);
    }

    __host__ __device__
    double toDouble() const {
        uint32_t sign = (value >> 7) & 0x1;
        int32_t mantissa = value & ((1u << NEFormat::MANTISSA_LENGTH) - 1);
        double value = (NEFormat::MANTISSA_FACTOR - NEMath::Log((float)mantissa)) * NEFormat::INV_LOG_BASE;
        if (sign) value = -value;
        return value;
    }

    __host__ __device__
    unsigned int toUint() const {
        uint32_t mantissa = value & NEFormat::MAX_MANTISSA;
        float value = (NEFormat::MANTISSA_FACTOR - NEMath::Log((float)mantissa)) * NEFormat::INV_LOG_BASE;
        return NEMath::NearbyUInt(value);  
    }

};
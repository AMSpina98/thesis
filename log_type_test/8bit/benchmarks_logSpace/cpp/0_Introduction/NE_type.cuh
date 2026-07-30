#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
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
    constexpr double BASE = 2.0; //just a reminder, never used: to chage base is mandatory to to work on the factor and the scale
    constexpr double LN2 = 0.6931471805599453;
    constexpr double MANTISSA_FACTOR = static_cast<double>(MANTISSA_LENGTH) * LN2;
    constexpr uint32_t MANTISSA_SCALE = 1u << MANTISSA_LENGTH;
    constexpr uint32_t MAX_MANTISSA = (1u << MANTISSA_LENGTH) - 1;
    constexpr float LOG_BASE = 0.6931471805599453f;
    constexpr float INV_LOG_BASE = 1.4426950408889634f;
    constexpr uint8_t NE_NAN = (1u << 7);
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

    __host__ __device__ inline float DExp(double x) {
    #ifdef __CUDA_ARCH__
        return exp(x);
    #else
        return std::exp(x);
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

    __host__ __device__ inline bool Isnan(float x) {
    #ifdef __CUDA_ARCH__
        return isnan(x);
    #else
        return std::isnan(x);
    #endif
    }

    __host__ __device__ inline NE dMul(NE A, NE B) {
        if(A == NEFormat::NE_NAN || B == NEFormat::NE_NAN) return NE(NEFormat::NE_NAN);
        uint8_t mantissaA = (A & NEFormat::MAX_MANTISSA);
        uint8_t mantissaB = (B & NEFormat::MAX_MANTISSA);
        if(mantissaA == 0 || mantissaB == 0) return NE(0);
        uint8_t signA = (A >> 7) & 0x1;
        uint8_t signB = (B >> 7) & 0x1;
        float xa = (NEMath::Log((float)mantissaA) - NEFormat::MANTISSA_FACTOR) * NEFormat::INV_LOG_BASE;
        float xb = (NEMath::Log((float)mantissaB) - NEFormat::MANTISSA_FACTOR) * NEFormat::INV_LOG_BASE;
        float xc = xa * xb;
        float I = NEMath::Exp(- xc * NEFormat::LOG_BASE) * NEFormat::MANTISSA_SCALE;
        uint8_t mantissaC = NEMath::NearbyUInt(I);
        if(mantissaC > NEFormat::MAX_MANTISSA) mantissaC = NEFormat::MAX_MANTISSA;
        return ((signA ^ signB) << 7) | mantissaC;
    }


    __host__ __device__ inline NE dDiv(NE A, NE B) {
        uint8_t mantissaA = (A & NEFormat::MAX_MANTISSA);
        uint8_t mantissaB = (B & NEFormat::MAX_MANTISSA);
        if(mantissaB == 0) return NE(NEFormat::NE_NAN);
        if(mantissaA == 0) return NE(0);
        uint8_t signA = (A >> 7) & 0x1;
        uint8_t signB = (B >> 7) & 0x1;
        float xa = (NEMath::Log((float)mantissaA) - NEFormat::MANTISSA_FACTOR) * NEFormat::INV_LOG_BASE;
        float xb = (NEMath::Log((float)mantissaB) - NEFormat::MANTISSA_FACTOR) * NEFormat::INV_LOG_BASE;
        float xc = xa / xb;
        float I = NEMath::Exp(- xc * NEFormat::LOG_BASE) * NEFormat::MANTISSA_SCALE;
        uint8_t mantissaC = NEMath::NearbyUInt(I);
        if(mantissaC > NEFormat::MAX_MANTISSA) mantissaC = NEFormat::MAX_MANTISSA;
        return ((signA ^ signB) << 7) | mantissaC;
    }

    __host__ __device__ inline NE dAdd(NE A, NE B) {
        uint8_t mantissaA = (A & NEFormat::MAX_MANTISSA);
        uint8_t mantissaB = (B & NEFormat::MAX_MANTISSA);
        uint8_t signA = (A >> 7) & 0x1;
        uint8_t signB = (B >> 7) & 0x1;
        if(mantissaA == 0) {
            if(signA == 1) return NE(NEFormat::NE_NAN);
            return B;
        }
        if(mantissaB == 0) {
            if(signB == 1) return NE(NEFormat::NE_NAN);
            return A;
        }
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
            return (signA << 7) | mantissa;
        } else{
            if (mantissaTempA == mantissaTempB) return NE(static_cast<uint8_t>(0));
            uint64_t mantissa = (((uint64_t)mantissaTempB << NEFormat::MANTISSA_LENGTH) / (uint64_t)mantissaTempA);
            return (signB << 7) | mantissa;
        }
    }


    __host__ __device__ inline NE dSub(NE A, NE B){
        return NEMath::dAdd(A, (B ^ (1u << 7)));
    }
}


//Converter
template<typename T>
struct Converter{
	__host__ __device__ static NE to(T x);
	__host__ __device__ static T from(NE x);
};


template<>
struct Converter<int> {
    __host__ __device__ static NE to(int x){
        if(x == 0) return uint8_t(0);
        uint32_t sign = (x < 0);
        float ax = NEMath::Abs(x);
        float I = NEMath::Exp(-ax * NEFormat::LOG_BASE);
        uint32_t mantissa = NEMath::NearbyUInt(I * NEFormat::MANTISSA_SCALE);
        if (mantissa > NEFormat::MAX_MANTISSA) mantissa = NEFormat::MAX_MANTISSA;
        return (sign << 7) |  mantissa;
    }

    __host__ __device__ static int from(NE x){
        if(x == 0) return 0;
        uint32_t sign = (x >> 7) & 0x1;
        int32_t mantissa = x & ((1u << NEFormat::MANTISSA_LENGTH) - 1);
        float value = (NEFormat::MANTISSA_FACTOR - NEMath::Log((float)mantissa)) * NEFormat::INV_LOG_BASE;
        if (sign) value = -value;
        return NEMath::NearbyInt(value);
    }
};


template<>
struct Converter<unsigned int> {
    __host__ __device__ static NE to(unsigned int x){
        if(x == 0) return uint8_t(0);
        float ax = (float)x;
        float I = NEMath::Exp(-ax * NEFormat::LOG_BASE);
        uint32_t mantissa = NEMath::NearbyUInt(I * NEFormat::MANTISSA_SCALE);
        mantissa = NEMath::Max(1u, mantissa);
        mantissa = NEMath::Min(NEFormat::MAX_MANTISSA, mantissa);
        return (0u << 7) | mantissa;
    }


    __host__ __device__ static unsigned int from(NE x){
        if(x == 0) return 0;
        uint32_t mantissa = x & NEFormat::MAX_MANTISSA;
        float value = (NEFormat::MANTISSA_FACTOR - NEMath::Log((float)mantissa)) * NEFormat::INV_LOG_BASE;
        return NEMath::NearbyUInt(value);
    }
};

template<>
struct Converter<double> {
    __host__ __device__ static NE to(double x){
        if(x == 0) return uint8_t(0);
        if(NEMath::Isnan(x)) return NEFormat::NE_NAN;
        uint32_t sign = (x < 0.0);
        double ax = NEMath::DAbs(x);
        double I = NEMath::DExp(-ax * NEFormat::LOG_BASE);
        uint32_t mantissa = NEMath::NearbyUInt(I * NEFormat::MANTISSA_SCALE);
        if (mantissa > NEFormat::MAX_MANTISSA) mantissa = NEFormat::MAX_MANTISSA;
        return (sign << 7) | mantissa;
    }

    __host__ __device__ static double from(NE x) {
        if(x == 0) return 0;
        if(x == NEFormat::NE_NAN) return NAN;
        uint32_t sign = (x >> 7) & 0x1;
        int32_t mantissa = x & ((1u << NEFormat::MANTISSA_LENGTH) - 1);
        double value = (NEFormat::MANTISSA_FACTOR - NEMath::Log((float)mantissa)) * NEFormat::INV_LOG_BASE;
        if (sign) value = -value;
        return value;
    }
};

template<>
struct Converter<float> {
	__host__ __device__ static NE to(float x){
        if(x == 0) return uint8_t(0);
        if(NEMath::Isnan(x)) return NEFormat::NE_NAN;
        uint32_t sign = (x < 0.0);
        float ax = NEMath::Abs(x);
        float I = NEMath::Exp(-ax * NEFormat::LOG_BASE);
        uint32_t mantissa = NEMath::NearbyUInt(I * NEFormat::MANTISSA_SCALE);
        if (mantissa > NEFormat::MAX_MANTISSA) mantissa = NEFormat::MAX_MANTISSA;
        return (sign << 7) | mantissa;
	}
	__host__ __device__ static float from(NE x){
        if(x == 0) return 0;
        if(x == NEFormat::NE_NAN) return NAN;
        uint32_t sign = (x >> 7) & 0x1;
        int32_t mantissa = x & ((1u << NEFormat::MANTISSA_LENGTH) - 1);
        float value = (NEFormat::MANTISSA_FACTOR - NEMath::Log((float)mantissa)) * NEFormat::INV_LOG_BASE;
        if (sign) value = -value;
        return value;
	}
};


struct NE {
    uint8_t value;

    __host__ __device__ NE() : value(0) {}
    __host__ __device__ explicit NE(uint8_t v) : value(v) {}
    __host__ __device__ explicit operator uint8_t() const { return value; }

    __host__ __device__ NE operator+(const NE& other) const {
        return NEMath::dAdd(*this, other);
    }

    __host__ __device__ NE operator-(const NE& other) const {
        return NEMath::dSub(*this, other);
    }

    __host__ __device__ NE operator*(const NE& other) const {
        return NEMath::dMul(*this, other);
    }

    __host__ __device__ NE operator/(const NE& other) const {
        return NEMath::dDiv(*this, other);
    }

    __host__ __device__ NE(float x) {
        value = Converter<float>::to(x);
    }

    __host__ __device__ operator float() const {
        return Converter<float>::from(*this);
    }

    __host__ __device__ operator int() const {
        return Converter<int>::from(*this);
    }

    __host__ __device__ operator double() const {
        return Converter<double>::from(*this);
    }

    __host__ __device__ operator unsigned int() const {
        return Converter<unsigned int>::from(*this);
    }
};



template<typename T> cudaError_t cudaMemcpyNE(
    NE* dst,
    const T* src,
    size_t count,
    cudaMemcpyKind kind){
    if(kind == cudaMemcpyHostToDevice){
        std::vector<NE> buffer(count);

        for(size_t i=0;i<count;i++){
            buffer[i] = Converter<T>::to(src[i]);
        }

        return cudaMemcpy(
            dst,
            buffer.data(),
            count*sizeof(NE),
            cudaMemcpyHostToDevice
        );
    }
    return cudaErrorInvalidValue;
}

template<typename T>
cudaError_t cudaMemcpyNEAsync(
    NE* dst,
    const T* src,
    size_t count,
    cudaMemcpyKind kind,
    cudaStream_t stream)
{
    if(kind != cudaMemcpyHostToDevice)
        return cudaErrorInvalidMemcpyDirection;

    NE* buffer;
    cudaError_t err =
        cudaMallocHost(&buffer, count*sizeof(NE));
    if(err != cudaSuccess)
        return err;
    for(size_t i=0;i<count;i++)
        buffer[i] = Converter<T>::to(src[i]);
    err = cudaMemcpyAsync(
        dst,
        buffer,
        count*sizeof(NE),
        cudaMemcpyHostToDevice,
        stream
    );
    if(err != cudaSuccess){
        cudaFreeHost(buffer);
        return err;
    }
    cudaStreamAddCallback(
        stream,
        [](cudaStream_t, cudaError_t, void* ptr){
            cudaFreeHost(ptr);
        },
        buffer,
        0
    );
    return cudaSuccess;
}

template<typename T>
__device__ void deviceConverter(NE* dst, const T* src, size_t count){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) dst[i] = Converter<T>::to(src[i]);
}

template<typename T>
__device__ void deviceTranslator(T* dst, const NE* src, size_t count){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) dst[i] = Converter<T>::from(src[i]);
}
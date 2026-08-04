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

struct NL;

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

}


struct NL {
    uint8_t value;

    __host__ __device__ NL() : value(0) {}
    __host__ __device__ NL(uint8_t v) : value(v) {}
    __host__ __device__ operator uint8_t() const { return value; }

    __host__ __device__ NL operator+(const NL& other) const {
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        uint8_t mantissaTempA = NLMath::Max(mantissaA, mantissaB);
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
            if(mantissa>NLFormat::MAX_MANTISSA) mantissa=NLFormat::MAX_MANTISSA;
            return NL((signA << 7) | mantissa);
        } else{
            uint64_t mantissa = (((uint64_t)mantissaTempB << NLFormat::MANTISSA_LENGTH) / (uint64_t)mantissaTempA);
            return NL((signB << 7) | mantissa);
        }
    }

    __host__ __device__ NL operator+=(const NL& other) const {
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        uint8_t mantissaTempA = NLMath::Max(mantissaA, mantissaB);
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
            if(mantissa>NLFormat::MAX_MANTISSA) mantissa=NLFormat::MAX_MANTISSA;
            return NL((signA << 7) | mantissa);
        } else{
            uint64_t mantissa = (((uint64_t)mantissaTempB << NLFormat::MANTISSA_LENGTH) / (uint64_t)mantissaTempA);
            return NL((signB << 7) | mantissa);
        }
    }

    __host__ __device__ NL operator-(const NL& other) const {
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = ((other.value >> 7) & 0x1) ^ 0x1;
        uint8_t mantissaTempA = NLMath::Max(mantissaA, mantissaB);
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
            if(mantissa>NLFormat::MAX_MANTISSA) mantissa=NLFormat::MAX_MANTISSA;
            return NL((signA << 7) | mantissa);
        } else{
            uint64_t mantissa = (((uint64_t)mantissaTempB << NLFormat::MANTISSA_LENGTH) / (uint64_t)mantissaTempA);
            return NL((signB << 7) | mantissa);
        }
    }

    __host__ __device__ NL operator-=(const NL& other) const {
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = ((other.value >> 7) & 0x1) ^ 0x1;
        uint8_t mantissaTempA = NLMath::Max(mantissaA, mantissaB);
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
            if(mantissa>NLFormat::MAX_MANTISSA) mantissa=NLFormat::MAX_MANTISSA;
            return NL((signA << 7) | mantissa);
        } else{
            uint64_t mantissa = (((uint64_t)mantissaTempB << NLFormat::MANTISSA_LENGTH) / (uint64_t)mantissaTempA);
            return NL((signB << 7) | mantissa);
        }
    }

    __host__ __device__ NL operator*(const NL& other) const {
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        float xa = (NLMath::Log((float)mantissaA) - NLFormat::MANTISSA_FACTOR) * NLFormat::INV_LOG_BASE;
        float xb = (NLMath::Log((float)mantissaB) - NLFormat::MANTISSA_FACTOR) * NLFormat::INV_LOG_BASE;
        float xc = xa * xb;
        float I = NLMath::Exp(- xc * NLFormat::LOG_BASE) * NLFormat::MANTISSA_SCALE;
        uint8_t mantissaC = NLMath::NearbyUInt(I);
        if(mantissaC > NLFormat::MAX_MANTISSA) mantissaC = NLFormat::MAX_MANTISSA;
        return NL(((signA ^ signB) << 7) | mantissaC);
    }

    __host__ __device__ NL operator*=(const NL& other) const {
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        float xa = (NLMath::Log((float)mantissaA) - NLFormat::MANTISSA_FACTOR) * NLFormat::INV_LOG_BASE;
        float xb = (NLMath::Log((float)mantissaB) - NLFormat::MANTISSA_FACTOR) * NLFormat::INV_LOG_BASE;
        float xc = xa * xb;
        float I = NLMath::Exp(- xc * NLFormat::LOG_BASE) * NLFormat::MANTISSA_SCALE;
        uint8_t mantissaC = NLMath::NearbyUInt(I);
        if(mantissaC > NLFormat::MAX_MANTISSA) mantissaC = NLFormat::MAX_MANTISSA;
        return NL(((signA ^ signB) << 7) | mantissaC);
    }

    __host__ __device__ NL operator/(const NL& other) const {
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        float xa = (NLMath::Log((float)mantissaA) - NLFormat::MANTISSA_FACTOR) * NLFormat::INV_LOG_BASE;
        float xb = (NLMath::Log((float)mantissaB) - NLFormat::MANTISSA_FACTOR) * NLFormat::INV_LOG_BASE;
        float xc = xa / xb;
        float I = NLMath::Exp(- xc * NLFormat::LOG_BASE) * NLFormat::MANTISSA_SCALE;
        uint8_t mantissaC = NLMath::NearbyUInt(I);
        if(mantissaC > NLFormat::MAX_MANTISSA) mantissaC = NLFormat::MAX_MANTISSA;
        return NL(((signA ^ signB) << 7) | mantissaC);
    }

    __host__ __device__ NL operator/=(const NL& other) const {
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        float xa = (NLMath::Log((float)mantissaA) - NLFormat::MANTISSA_FACTOR) * NLFormat::INV_LOG_BASE;
        float xb = (NLMath::Log((float)mantissaB) - NLFormat::MANTISSA_FACTOR) * NLFormat::INV_LOG_BASE;
        float xc = xa / xb;
        float I = NLMath::Exp(- xc * NLFormat::LOG_BASE) * NLFormat::MANTISSA_SCALE;
        uint8_t mantissaC = NLMath::NearbyUInt(I);
        if(mantissaC > NLFormat::MAX_MANTISSA) mantissaC = NLFormat::MAX_MANTISSA;
        return NL(((signA ^ signB) << 7) | mantissaC);
    }
};



//Converter
template<typename T>
struct Converter{
	__host__ __device__ static NL to(T x);
	__host__ __device__ static T from(NL x);
};


template<>
struct Converter<int> {
    __host__ __device__ static NL to(int x){
        uint32_t sign = (x < 0);
        float ax = NLMath::Abs(x);
        float I = NLMath::Exp(-ax * NLFormat::LOG_BASE);
        uint32_t mantissa = NLMath::NearbyUInt(I * NLFormat::MANTISSA_SCALE);
        if (mantissa > NLFormat::MAX_MANTISSA) mantissa = NLFormat::MAX_MANTISSA;
        return NL((sign << 7) |  mantissa);
    }

    __host__ __device__ static int from(NL x){
        uint32_t sign = (x >> 7) & 0x1;
        int32_t mantissa = x & ((1u << NLFormat::MANTISSA_LENGTH) - 1);
        float value = (NLFormat::MANTISSA_FACTOR - NLMath::Log((float)mantissa)) * NLFormat::INV_LOG_BASE;
        if (sign) value = -value;
        return NLMath::NearbyInt(value);
    }
};


template<>
struct Converter<unsigned int> {
    __host__ __device__ static NL to(unsigned int x){
        float ax = (float)x;
        float I = NLMath::Exp(-ax * NLFormat::LOG_BASE);
        uint32_t mantissa = NLMath::NearbyUInt(I * NLFormat::MANTISSA_SCALE);
        mantissa = NLMath::Max(1u, mantissa);
        mantissa = NLMath::Min(NLFormat::MAX_MANTISSA, mantissa);
        return NL((0u << 7) | mantissa);
    }


    __host__ __device__ static unsigned int from(NL x){
        uint32_t mantissa = x & NLFormat::MAX_MANTISSA;
        float value = (NLFormat::MANTISSA_FACTOR - NLMath::Log((float)mantissa)) * NLFormat::INV_LOG_BASE;
        return NLMath::NearbyUInt(value);
    }
};

template<>
struct Converter<double> {
    __host__ __device__ static NL to(double x){
        uint32_t sign = (x < 0.0);
        double ax = NLMath::DAbs(x);
        double I = NLMath::Exp(-ax * NLFormat::LOG_BASE);
        uint32_t mantissa = NLMath::NearbyUInt(I * NLFormat::MANTISSA_SCALE);
        if (mantissa > NLFormat::MAX_MANTISSA) mantissa = NLFormat::MAX_MANTISSA;
        return NL((sign << 7) | mantissa);
    }

    __host__ __device__ static double from(NL x) {
        uint32_t sign = (x >> 7) & 0x1;
        int32_t mantissa = x & ((1u << NLFormat::MANTISSA_LENGTH) - 1);
        double value = (NLFormat::MANTISSA_FACTOR - NLMath::Log((float)mantissa)) * NLFormat::INV_LOG_BASE;
        if (sign) value = -value;
        return value;
    }
};

template<>
struct Converter<float> {
	__host__ __device__ static NL to(float x){
        uint32_t sign = (x < 0.0);
        float ax = NLMath::Abs(x);
        float I = NLMath::Exp(-ax * NLFormat::LOG_BASE);
        uint32_t mantissa = NLMath::NearbyUInt(I * NLFormat::MANTISSA_SCALE);
        if (mantissa > NLFormat::MAX_MANTISSA) mantissa = NLFormat::MAX_MANTISSA;
        return NL((sign << 7) | mantissa);
	}
	__host__ __device__ static float from(NL x){
        uint32_t sign = (x >> 7) & 0x1;
        int32_t mantissa = x & ((1u << NLFormat::MANTISSA_LENGTH) - 1);
        float value = (NLFormat::MANTISSA_FACTOR - NLMath::Log((float)mantissa)) * NLFormat::INV_LOG_BASE;
        if (sign) value = -value;
        return value;
	}
};

/*
template<typename T> cudaError_t cudaMemcpyNL(
    NL* dst,
    const T* src,
    size_t count,
    cudaMemcpyKind kind){
    if(kind == cudaMemcpyHostToDevice){
        std::vector<NL> buffer(count);

        for(size_t i=0;i<count;i++){
            buffer[i] = Converter<T>::to(src[i]);
        }

        return cudaMemcpy(
            dst,
            buffer.data(),
            count*sizeof(NL),
            cudaMemcpyHostToDevice
        );
    }
    return cudaErrorInvalidValue;
}


template<typename T>
cudaError_t cudaMemcpyNLAsync(
    NL* dst,
    const T* src,
    size_t count,
    cudaMemcpyKind kind,
    cudaStream_t stream)
{
    if(kind != cudaMemcpyHostToDevice)
        return cudaErrorInvalidMemcpyDirection;

    NL* buffer;
    cudaError_t err =
        cudaMallocHost(&buffer, count*sizeof(NL));
    if(err != cudaSuccess)
        return err;
    for(size_t i=0;i<count;i++)
        buffer[i] = Converter<T>::to(src[i]);
    err = cudaMemcpyAsync(
        dst,
        buffer,
        count*sizeof(NL),
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
__device__ void deviceConverter(NL* dst, const T* src, size_t count){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) dst[i] = Converter<T>::to(src[i]);
}

template<typename T>
__device__ void deviceTranslator(T* dst, const NL* src, size_t count){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) dst[i] = Converter<T>::from(src[i]);
}

*/
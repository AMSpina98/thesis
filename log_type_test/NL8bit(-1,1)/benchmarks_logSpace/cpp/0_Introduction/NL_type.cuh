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

}


struct NL {
    uint8_t value;

    __host__ __device__ NL() : value(0) {}
    __host__ __device__ NL(uint8_t v) : value(v) {}
    __host__ __device__ NL(float x);
    __host__ __device__ NL(double x);
    __host__ __device__ operator uint8_t() const { return value; }

    __host__ __device__ NL operator+(const NL& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return NL(static_cast<uint8_t>(LOG_NAN));
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
            if(xa == xb) return NL(static_cast<uint8_t>(LOG_ZERO));
            xc = -NLMath::Log(NLMath::Abs(NLMath::Exp(-xa) - NLMath::Exp(-xb)));
            if(xa > xb) signC = signA;
            else signC = signB;
        }
        if(xc < 0.0f) xc = 0.0f;
        uint8_t mantissaC = NLMath::NearbyUInt(xc * 8.0f);
        if(mantissaC > NLFormat::MAX_MANTISSA) return NL(static_cast<uint8_t>(LOG_ZERO));
        return NL(static_cast<uint8_t>((signC << 7) | mantissaC));
    }

    __host__ __device__ NL operator+=(const NL& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return NL(static_cast<uint8_t>(LOG_NAN));
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
            if(xa == xb) return NL(static_cast<uint8_t>(LOG_ZERO));
            xc = -NLMath::Log(NLMath::Abs(NLMath::Exp(-xa) - NLMath::Exp(-xb)));
            if(xa > xb) signC = signA;
            else signC = signB;
        }
        if(xc < 0.0f) xc = 0.0f;
        uint8_t mantissaC = NLMath::NearbyUInt(xc * 8.0f);
        if(mantissaC > NLFormat::MAX_MANTISSA) return NL(static_cast<uint8_t>(LOG_ZERO));
        return NL(static_cast<uint8_t>((signC << 7) | mantissaC));
    }

    __host__ __device__ NL operator-(const NL& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return NL(static_cast<uint8_t>(LOG_NAN));
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
            if(xa == xb) return NL(static_cast<uint8_t>(LOG_ZERO));
            xc = -NLMath::Log(NLMath::Abs(NLMath::Exp(-xa) + NLMath::Exp(-xb)));
            if(xa > xb) signC = signA;
            else signC = signB;
        }
        if(xc < 0.0f) xc = 0.0f;
        uint8_t mantissaC = NLMath::NearbyUInt(xc * 8.0f);
        if(mantissaC > NLFormat::MAX_MANTISSA) return NL(static_cast<uint8_t>(LOG_ZERO));
        return NL(static_cast<uint8_t>((signC << 7) | mantissaC));
    }

    __host__ __device__ NL operator-=(const NL& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return NL(static_cast<uint8_t>(LOG_NAN));
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
            if(xa == xb) return NL(static_cast<uint8_t>(LOG_ZERO));
            xc = -NLMath::Log(NLMath::Abs(NLMath::Exp(-xa) + NLMath::Exp(-xb)));
            if(xa > xb) signC = signA;
            else signC = signB;
        }
        if(xc < 0.0f) xc = 0.0f;
        uint8_t mantissaC = NLMath::NearbyUInt(xc * 8.0f);
        if(mantissaC > NLFormat::MAX_MANTISSA) return NL(static_cast<uint8_t>(LOG_ZERO));
        return NL(static_cast<uint8_t>((signC << 7) | mantissaC));
    }

    __host__ __device__ NL operator*(const NL& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return NL(static_cast<uint8_t>(LOG_NAN));
        if(value == LOG_ZERO || other.value == LOG_ZERO) return NL(static_cast<uint8_t>(LOG_ZERO));
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        uint8_t mantissaC = mantissaA + mantissaB;
        if(mantissaC > NLFormat::MAX_MANTISSA) mantissaC = NLFormat::MAX_MANTISSA;
        return NL(static_cast<uint8_t>(((signA ^ signB) << 7) | mantissaC));
    }

    __host__ __device__ NL operator*=(const NL& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return NL(static_cast<uint8_t>(LOG_NAN));
        if(value == LOG_ZERO || other.value == LOG_ZERO) return NL(static_cast<uint8_t>(LOG_ZERO));
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        uint8_t mantissaC = mantissaA + mantissaB;
        if(mantissaC > NLFormat::MAX_MANTISSA) mantissaC = NLFormat::MAX_MANTISSA;
        return NL(static_cast<uint8_t>(((signA ^ signB) << 7) | mantissaC));
    }

    __host__ __device__ NL operator/(const NL& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return NL(static_cast<uint8_t>(LOG_NAN));
        if(value == LOG_ZERO) return NL(static_cast<uint8_t>(LOG_ZERO));
        if(other.value == LOG_ZERO) return NL(static_cast<uint8_t>(LOG_NAN));
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        uint8_t mantissaC = mantissaA > mantissaB ? mantissaA - mantissaB : 0;
        if(mantissaC > NLFormat::MAX_MANTISSA) mantissaC = NLFormat::MAX_MANTISSA;
        return NL(static_cast<uint8_t>(((signA ^ signB) << 7) | mantissaC));
    }

    __host__ __device__ NL operator/=(const NL& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return NL(static_cast<uint8_t>(LOG_NAN));
        if(value == LOG_ZERO) return NL(static_cast<uint8_t>(LOG_ZERO));
        if(other.value == LOG_ZERO) return NL(static_cast<uint8_t>(LOG_NAN));
        uint8_t mantissaA = (value & NLFormat::MAX_MANTISSA);
        uint8_t mantissaB = (other.value & NLFormat::MAX_MANTISSA);
        uint8_t signA = (value >> 7) & 0x1;
        uint8_t signB = (other.value >> 7) & 0x1;
        uint8_t mantissaC = mantissaA > mantissaB ? mantissaA - mantissaB : 0;
        if(mantissaC > NLFormat::MAX_MANTISSA) mantissaC = NLFormat::MAX_MANTISSA;
        return NL(static_cast<uint8_t>(((signA ^ signB) << 7) | mantissaC));
    }
};



//Converter
template<typename T>
struct Converter{
	__host__ __device__ static NL to(T x);
	__host__ __device__ static T from(NL x);
};


template<>
struct Converter<double> {
    __host__ __device__ static NL to(double x){
        if(x == 0.0) return NL(static_cast<uint8_t>(LOG_ZERO));
        if(std::isnan(x)) return NL(static_cast<uint8_t>(LOG_NAN));
        uint8_t sign = (x < 0.0) ? 1 : 0;
        double y = - NLMath::Log(NLMath::Abs(x));
        uint8_t mantissa = NLMath::NearbyUInt(y * 8.0);
        if(mantissa > NLFormat::MAX_MANTISSA) mantissa = NLFormat::MAX_MANTISSA;
        return NL(static_cast<uint8_t>((sign << 7) | mantissa));
    }

    __host__ __device__ static double from(NL x) {
        if(x.value == LOG_ZERO) return 0.0;
        if(x.value == LOG_NAN) return std::numeric_limits<double>::quiet_NaN();
        uint8_t mantissa = (x.value & NLFormat::MAX_MANTISSA);
        uint8_t sign = (x.value >> 7) & 0x1;
        double y = ((double)mantissa) / 8.0;
        double result = NLMath::Exp(-y);
        return (sign == 1) ? -result : result;
    }
};

template<>
struct Converter<float> {
	__host__ __device__ static NL to(float x){
        if(x == 0.0) return NL(static_cast<uint8_t>(LOG_ZERO));
        if(std::isnan(x)) return NL(static_cast<uint8_t>(LOG_NAN));
        uint8_t sign = (x < 0.0) ? 1 : 0;
        float y = - NLMath::Log(NLMath::Abs(x));
        uint8_t mantissa = NLMath::NearbyUInt(y * 8.0);
        if(mantissa > NLFormat::MAX_MANTISSA) mantissa = NLFormat::MAX_MANTISSA;
        return NL(static_cast<uint8_t>((sign << 7) | mantissa));
	}
	__host__ __device__ static float from(NL x){
        if(x.value == LOG_ZERO) return 0.0f;
        if(x.value == LOG_NAN) return std::numeric_limits<float>::quiet_NaN();
	    uint8_t mantissa = (x.value & NLFormat::MAX_MANTISSA);
	    uint8_t sign = (x.value >> 7) & 0x1;
	    float y = ((float)mantissa) / 8.0;
	    float result = NLMath::Exp(-y);
	    return (sign == 1) ? -result : result;
	}
};

__host__ __device__
NL::NL(float x) {
    *this = Converter<float>::to(x);
}

__host__ __device__
NL::NL(double x) {
    *this = Converter<double>::to(x);
}

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
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

template<typename T>
struct Encapsulator;

namespace LNNSMath {

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

template<typename T>
struct Encapsulator {
    T value;
    __host__ __device__
    Encapsulator() = default;
    __host__ __device__
    Encapsulator(T v) : value(v) {}
    __host__ __device__
    operator T() const { return value; }
    
    public:
    __host__ __device__ Encapsulator(float x){ 
        value = static_cast<T>(x);
    }
    __host__ __device__ Encapsulator(double x){ 
        value = static_cast<T>(x);
    }

    __host__ __device__ Encapsulator(unsigned int x){
        value = static_cast<T>(x);
    }

    __host__ __device__ Encapsulator(int x){
        value = static_cast<T>(x);
    }

    __host__ __device__ LNNS operator+(const LNNS& other) const {
        float a = toFloat();
        float b = other.toFloat();
        float c = max(fabs(a), fabs(b));
        if(c < 1.0f) c = 1.0f;
        a = dDiv(a,c);
        b = dDiv(b,c);
        float result = __logf(__expf(a) * __expf(b));
        result = dMul(result, c);
        return Converter<T>::from(result);
    }

    __host__ __device__ LNNS& operator+=(const LNNS& other) {
        
    }

    __host__ __device__ LNNS operator-(const LNNS& other) const {
            
    }

    __host__ __device__ LNNS& operator-=(const LNNS& other) {
        
    }

    __host__ __device__ LNNS operator*(const LNNS& other) const {
        float a = toFloat();
        float b = other.toFloat();
        bool sign = (a < 0.0f) ^ (b < 0.0f);
        float ma, mb;
        int ea, eb;
        ma = frexpf(fabsf(a), &ea);
        mb = frexpf(fabsf(b), &eb);
        float result = __expf(__logf(fabsf(ma)) + __logf(fabsf(mb)));
        result = ldexpf(result, (ea + eb));
        if (sign) result = -result;
        return Converter<T>::from(result);        
    }

    __host__ __device__ LNNS& operator*=(const LNNS& other) {
        float a = toFloat();
        float b = other.toFloat();
        bool sign = (a < 0.0f) ^ (b < 0.0f);
        float ma, mb;
        int ea, eb;
        ma = frexpf(fabsf(a), &ea);
        mb = frexpf(fabsf(b), &eb);
        float result = __expf(__logf(fabsf(ma)) + __logf(fabsf(mb)));
        result = ldexpf(result, (ea + eb));
        if (sign) result = -result;
        value = Converter<T>::from(result);
        return *this;
    }

    __host__ __device__ LNNS operator/(const LNNS& other) const {
        float a = toFloat();
        float b = other.toFloat();
        bool sign = (a < 0.0f) ^ (b < 0.0f);
        float ma, mb;
        int ea, eb;
        ma = frexpf(fabsf(a), &ea);
        mb = frexpf(fabsf(b), &eb);
        float result = __expf(__logf(fabsf(ma)) - __logf(fabsf(mb)));
        result = ldexpf(result, (ea - eb));
        if (sign) result = -result;
        return Converter<T>::from(result);
    }

    __host__ __device__ LNNS& operator/=(const LNNS& other) {
        float a = toFloat();
        float b = other.toFloat();
        bool sign = (a < 0.0f) ^ (b < 0.0f);
        float ma, mb;
        int ea, eb;
        ma = frexpf(fabsf(a), &ea);
        mb = frexpf(fabsf(b), &eb);
        float result = __expf(__logf(fabsf(ma)) - __logf(fabsf(mb)));
        result = ldexpf(result, (ea - eb));
        if (sign) result = -result;
        value = Converter<T>::from(result);
        return *this;
    }

    __host__ __device__ bool operator<(const LNNS& other) const {
        
    }

    __host__ __device__ bool operator<=(const LNNS& other) const {
        
    }

    __host__ __device__ bool operator>(const LNNS& other) const {
        
    }

    __host__ __device__ bool operator>=(const LNNS& other) const {
        
    }

    __host__ __device__ bool operator==(const LNNS& other) const {
        
    }

    __host__ __device__ bool operator!=(const LNNS& other) const {
        
    }


    __host__ __device__
    float toFloat() const {
        return static_cast<float>(value); 
    }

    __host__ __device__
    double toDouble() const {
        return static_cast<double>(value);
    }

    __host__ __device__
    int toInt() const {
        return static_cast<int>(value);
    }

    __host__ __device__
    unsigned int toUInt() const {
        return static_cast<unsigned int>(value);
    }
};
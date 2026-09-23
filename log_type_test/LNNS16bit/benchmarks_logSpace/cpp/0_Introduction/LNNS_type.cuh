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

struct LNNS;

namespace LNNSFormat {

    constexpr int SIGN_BITS = 1;
    constexpr int MANTISSA_LENGTH = 15;
    constexpr uint32_t MAX_MANTISSA = (1u << MANTISSA_LENGTH) - 1;
    #define LOG_ZERO 0x4000
    #define LOG_NAN  0xc000
}

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


struct LNNS {
    uint16_t value;

    public:
    __host__ __device__
    static LNNS raw(uint16_t v) {
        return LNNS(v, RawTag{});
    }

    private:
    struct RawTag {};

    __host__ __device__
    LNNS(uint16_t v, RawTag) : value(v) {}
    
    public:
    __host__ __device__ LNNS(float x){   
        if(x == 0.0){
            value = LOG_ZERO;
            return;
        }
        if(LNNSMath::IsNaN(x)){
            value = LOG_NAN;
            return;
        }
        uint8_t sign = (x < 0.0) ? 1 : 0;
        float y = LNNSMath::Log(LNNSMath::Abs(x));
        int16_t mantissa = LNNSMath::NearbyInt(y * 1024.0f);
        if(mantissa > 32767) mantissa = 32767;
        if(mantissa < -32768) mantissa = -32768;
        uint16_t z = static_cast<uint16_t>(mantissa) & 0x7FFF;
        value = (sign << 15) | z;
    }
    __host__ __device__ LNNS(double x){  
        if(x == 0.0){
            value = LOG_ZERO;
            return;
        }
        if(LNNSMath::IsNaNDouble(x)){
            value = LOG_NAN;
            return;
        }
        uint8_t sign = (x < 0.0) ? 1 : 0;
        double y = LNNSMath::Log(LNNSMath::Abs(x));
        int16_t mantissa = LNNSMath::NearbyInt(y * 1024.0f);
        if(mantissa > 32767) mantissa = 32767;
        if(mantissa < -32768) mantissa = -32768;
        uint16_t z = static_cast<uint16_t>(mantissa) & 0x7FFF;
        value = (sign << 15) | z;
    }

    __host__ __device__ LNNS(unsigned int x){
        if(x == 0.0){
            value = LOG_ZERO;
            return;
        }
        if(LNNSMath::IsNaNDouble(x)){
            value = LOG_NAN;
            return;
        }
        float y = LNNSMath::Log(LNNSMath::Abs(x));
        int16_t mantissa = LNNSMath::NearbyInt(y * 1024.0f);
        if(mantissa > 32767) mantissa = 32767;
        if(mantissa < -32768) mantissa = -32768;
        value = static_cast<uint16_t>(mantissa) & 0x7FFF;
    }

    __host__ __device__ LNNS(int x){
        if(x == 0.0){
            value = LOG_ZERO;
            return;
        }
        if(LNNSMath::IsNaNDouble(x)){
            value = LOG_NAN;
            return;
        }
        uint8_t sign = (x < 0.0) ? 1 : 0;
        float y = LNNSMath::Log(LNNSMath::Abs(x));
        int16_t mantissa = LNNSMath::NearbyInt(y * 1024.0f);
        if(mantissa > 32767) mantissa = 32767;
        if(mantissa < -32768) mantissa = -32768;
        uint16_t z = static_cast<uint16_t>(mantissa) & 0x7FFF;
        value = (sign << 15) | z;
    }


    LNNS() = default;
    __host__ __device__ operator uint16_t() const { return value; }
    

    __host__ __device__ LNNS operator+(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return LNNS::raw(static_cast<uint16_t>(LOG_NAN));
        if(value == LOG_ZERO) return other;
        if(other.value == LOG_ZERO) return *this;
        int16_t mantissaA = ((value & 0x4000)<<1) | (value & 0x7fff);
        int16_t mantissaB = ((other.value & 0x4000)<<1) | (other.value & 0x7fff);
        uint16_t signA = (value >> 15) & 0x1;
        uint16_t signB = (other.value >> 15) & 0x1;
        float ya = ((float)mantissaA)/1024.0f;
        float yb = ((float)mantissaB)/1024.0f;
        if(ya < yb){
            std::swap(ya, yb);
            std::swap(signA, signB);
        }
        float yc;
        if(signA == signB){
            //xc = LNNSMath::Log(LNNSMath::Exp(xa) + LNNSMath::Exp(xb));
            yc = ya + LNNSMath::Log(1.0f + LNNSMath::Exp(yb - ya));
        }else{
            if(ya == yb) return LNNS::raw(static_cast<uint16_t>(LOG_ZERO));
            yc = ya + LNNSMath::Log(1.0f - LNNSMath::Exp(yb - ya));
        }
        int16_t mantissaC = (LNNSMath::NearbyInt(yc * 1024.0f));
        if(mantissaC > 32767) mantissaC = 32767;
        if(mantissaC < -32768) mantissaC = -32768;
        return LNNS::raw(static_cast<uint16_t>((signA << 15) | (mantissaC & 0x7fff)));
    }

    __host__ __device__ LNNS& operator+=(const LNNS& other) {
        if(value == LOG_NAN || other.value == LOG_NAN){
            value = LOG_NAN;
            return *this;
        }
        if(value == LOG_ZERO){
            value = other.value;
            return *this;
        }
        if(other.value == LOG_ZERO) return *this;
        int16_t mantissaA = ((value & 0x4000)<<1) | (value & 0x7fff);
        int16_t mantissaB = ((other.value & 0x4000)<<1) | (other.value & 0x7fff);
        uint16_t signA = (value >> 15) & 0x1;
        uint16_t signB = (other.value >> 15) & 0x1;
        float ya = ((float)mantissaA)/1024.0f;
        float yb = ((float)mantissaB)/1024.0f;
        if(ya < yb){
            std::swap(ya, yb);
            std::swap(signA, signB);
        }
        float yc;
        if(signA == signB){
            //xc = LNNSMath::Log(LNNSMath::Exp(xa) + LNNSMath::Exp(xb));
            yc = ya + LNNSMath::Log(1.0f + LNNSMath::Exp(yb - ya));
        }else{
            if(ya == yb){
                value = LOG_ZERO;
                return *this;
            }
            yc = ya + LNNSMath::Log(1.0f - LNNSMath::Exp(yb - ya));
        }
        int16_t mantissaC = (LNNSMath::NearbyInt(yc * 1024.0f));
        if(mantissaC > 32767) mantissaC = 32767;
        if(mantissaC < -32768) mantissaC = -32768;
        value = (signA << 15) | (mantissaC & 0x7fff);
        return *this;
    }

    __host__ __device__ LNNS operator-(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return LNNS::raw(static_cast<uint16_t>(LOG_NAN));
        if(value == LOG_ZERO) return LNNS::raw(0x8000 ^ other.value);
        if(other.value == LOG_ZERO) return *this;
        int16_t mantissaA = ((value & 0x4000)<<1) | (value & 0x7fff);
        int16_t mantissaB = ((other.value & 0x4000)<<1) | (other.value & 0x7fff);
        uint16_t signA = (value >> 15) & 0x1;
        uint16_t signB = (other.value >> 15) & 0x1;
        float ya = ((float)mantissaA)/1024.0f;
        float yb = ((float)mantissaB)/1024.0f;
        if(ya < yb){
            std::swap(ya, yb);
            std::swap(signA, signB);
        }
        float yc;
        if(signA != signB){
            yc = ya + LNNSMath::Log(1.0f + LNNSMath::Exp(yb - ya));
        }else{
            if(ya == yb) return LNNS::raw(static_cast<uint16_t>(LOG_ZERO));
            yc = ya + LNNSMath::Log(1.0f - LNNSMath::Exp(yb - ya));
        }
        int16_t mantissaC = (LNNSMath::NearbyInt(yc * 1024.0f));
        if(mantissaC > 32767) mantissaC = 32767;
        if(mantissaC < -32768) mantissaC = -32768;
        return LNNS::raw(static_cast<uint16_t>((signA << 15) | (mantissaC & 0x7fff)));
    }

    __host__ __device__ LNNS& operator-=(const LNNS& other) {
        if(value == LOG_NAN || other.value == LOG_NAN) {
            value = LOG_NAN;
            return *this;
        }
        if(value == LOG_ZERO) {
            value = 0x8000 ^ other.value;
            return *this;
        }
        if(other.value == LOG_ZERO) return *this;
        int16_t mantissaA = ((value & 0x4000)<<1) | (value & 0x7fff);
        int16_t mantissaB = ((other.value & 0x4000)<<1) | (other.value & 0x7fff);
        uint16_t signA = (value >> 15) & 0x1;
        uint16_t signB = (other.value >> 15) & 0x1;
        float ya = ((float)mantissaA)/1024.0f;
        float yb = ((float)mantissaB)/1024.0f;
        if(ya < yb){
            std::swap(ya, yb);
            std::swap(signA, signB);
        }
        float yc;
        if(signA != signB){
            yc = ya + LNNSMath::Log(1.0f + LNNSMath::Exp(yb - ya));
        }else{
            if(ya == yb){
                value = LOG_ZERO;
                return *this;
            }
            yc = ya + LNNSMath::Log(1.0f - LNNSMath::Exp(yb - ya));
        }
        int16_t mantissaC = (LNNSMath::NearbyInt(yc * 1024.0f));
        if(mantissaC > 32767) mantissaC = 32767;
        if(mantissaC < -32768) mantissaC = -32768;
        value = (signA << 15) | (mantissaC & 0x7fff);
        return *this;
    }

    __host__ __device__ LNNS operator*(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return LNNS::raw(static_cast<uint16_t>(LOG_NAN));
        if(value == LOG_ZERO || other.value == LOG_ZERO) return LNNS::raw(static_cast<uint16_t>(LOG_ZERO));
        int16_t mantissaA = ((value & 0x4000)<<1) | (value & 0x7fff);
        int16_t mantissaB = ((other.value & 0x4000)<<1) | (other.value & 0x7fff);
        uint16_t signA = (value >> 15) & 0x1;
        uint16_t signB = (other.value >> 15) & 0x1;
        int16_t mantissaC = mantissaA + mantissaB;
        if(mantissaC > 32767) mantissaC = 32767;
        if(mantissaC < -32768) mantissaC = -32768;
        return LNNS::raw(static_cast<uint16_t>(((signA ^ signB)<<15) | (mantissaC & 0x7fff)));
    }

    __host__ __device__ LNNS& operator*=(const LNNS& other) {
        if(value == LOG_NAN || other.value == LOG_NAN){
            value = LOG_NAN;
            return *this;
        }
        if(value == LOG_ZERO || other.value == LOG_ZERO){
            value = LOG_ZERO;
            return *this;
        }
        int16_t mantissaA = ((value & 0x4000)<<1) | (value & 0x7fff);
        int16_t mantissaB = ((other.value & 0x4000)<<1) | (other.value & 0x7fff);
        uint16_t signA = (value >> 15) & 0x1;
        uint16_t signB = (other.value >> 15) & 0x1;
        int16_t mantissaC = mantissaA + mantissaB;
        if(mantissaC > 32767) mantissaC = 32767;
        if(mantissaC < -32768) mantissaC = -32768;
        value = (signA ^ signB) << 15 | (mantissaC & 0x7fff);
        return *this;
    }

    __host__ __device__ LNNS operator/(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN || other.value == LOG_ZERO) return LNNS::raw(static_cast<uint16_t>(LOG_NAN));
        if(value == LOG_ZERO) return LNNS::raw(static_cast<uint16_t>(LOG_ZERO));
        int16_t mantissaA = ((value & 0x4000)<<1) | (value & 0x7fff);
        int16_t mantissaB = ((other.value & 0x4000)<<1) | (other.value & 0x7fff);
        uint16_t signA = (value >> 15) & 0x1;
        uint16_t signB = (other.value >> 15) & 0x1;
        int16_t mantissaC = mantissaA - mantissaB;
        if(mantissaC > 32767) mantissaC = 32767;
        if(mantissaC < -32768) mantissaC = -32768;
        return LNNS::raw(static_cast<uint16_t>(((signA ^ signB)<<15) | (mantissaC & 0x7fff)));
    }

    __host__ __device__ LNNS& operator/=(const LNNS& other) {
        if(value == LOG_NAN || other.value == LOG_NAN || other.value == LOG_ZERO) {
            value = LOG_NAN;
            return *this;
        }
        if(value == LOG_ZERO) {
            value = LOG_ZERO;
            return *this;
        }
        int16_t mantissaA = ((value & 0x4000)<<1) | (value & 0x7fff);
        int16_t mantissaB = ((other.value & 0x4000)<<1) | (other.value & 0x7fff);
        uint16_t signA = (value >> 15) & 0x1;
        uint16_t signB = (other.value >> 15) & 0x1;
        int16_t mantissaC = mantissaA - mantissaB;
        if(mantissaC > 32767) mantissaC = 32767;
        if(mantissaC < -32768) mantissaC = -32768;
        value = (signA ^ signB) << 15 | (mantissaC & 0x7fff);
        return *this;
    }

    __host__ __device__ bool operator<(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return false;
        int16_t mantissaA = ((value & 0x4000)<<1) | (value & 0x7fff);
        int16_t mantissaB = ((other.value & 0x4000)<<1) | (other.value & 0x7fff);
        uint16_t signA = (value >> 15) & 0x1;
        uint16_t signB = (other.value >> 15) & 0x1;
        if(signA != signB) return signA;
        if(signA == 0) return mantissaA < mantissaB;
        else return mantissaA > mantissaB;
    }

    __host__ __device__ bool operator<=(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return false;
        int16_t mantissaA = ((value & 0x4000)<<1) | (value & 0x7fff);
        int16_t mantissaB = ((other.value & 0x4000)<<1) | (other.value & 0x7fff);
        uint16_t signA = (value >> 15) & 0x1;
        uint16_t signB = (other.value >> 15) & 0x1;
        if(signA != signB) return signA;
        if(signA == 0) return mantissaA <= mantissaB;
        else return mantissaA >= mantissaB;
    }

    __host__ __device__ bool operator>(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return false;
        int16_t mantissaA = ((value & 0x4000)<<1) | (value & 0x7fff);
        int16_t mantissaB = ((other.value & 0x4000)<<1) | (other.value & 0x7fff);
        uint16_t signA = (value >> 15) & 0x1;
        uint16_t signB = (other.value >> 15) & 0x1;
        if(signA != signB) return signB;
        if(signA == 0) return mantissaA > mantissaB;
        else return mantissaA < mantissaB;
    }

    __host__ __device__ bool operator>=(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return false;
        int16_t mantissaA = ((value & 0x4000)<<1) | (value & 0x7fff);
        int16_t mantissaB = ((other.value & 0x4000)<<1) | (other.value & 0x7fff);
        uint16_t signA = (value >> 15) & 0x1;
        uint16_t signB = (other.value >> 15) & 0x1;
        if(signA != signB) return signB;
        if(signA == 0) return mantissaA >= mantissaB;
        else return mantissaA <= mantissaB;
    }

    __host__ __device__ bool operator==(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return false;
        int16_t mantissaA = ((value & 0x4000)<<1) | (value & 0x7fff);
        int16_t mantissaB = ((other.value & 0x4000)<<1) | (other.value & 0x7fff);
        uint16_t signA = (value >> 15) & 0x1;
        uint16_t signB = (other.value >> 15) & 0x1;
        if(signA != signB) return false;
        return mantissaA == mantissaB;
    }

    __host__ __device__ bool operator!=(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return true;
        int16_t mantissaA = ((value & 0x4000)<<1) | (value & 0x7fff);
        int16_t mantissaB = ((other.value & 0x4000)<<1) | (other.value & 0x7fff);
        uint16_t signA = (value >> 15) & 0x1;
        uint16_t signB = (other.value >> 15) & 0x1;
        if(signA != signB) return true;
        return mantissaA != mantissaB;
    }


    __host__ __device__
    float toFloat() const {
        if(value == LOG_ZERO) return 0.0f;
        if(value == LOG_NAN) return LNNSMath::Quiet_NAN();
        int16_t mantissa = ((value & 0x4000)<<1) | (value & 0x7fff);
	    uint16_t sign = (value >> 15) & 0x1;
	    float y = ((float)mantissa) / 8.0;
	    float result = LNNSMath::Exp(y);
	    return (sign == 1) ? -result : result;   
    }

    __host__ __device__
    double toDouble() const {
        if(value == LOG_ZERO) return 0.0;
        if(value == LOG_NAN) return LNNSMath::Quiet_NAN();
        int16_t mantissa = ((value & 0x4000)<<1) | (value & 0x7fff);
        uint16_t sign = (value >> 15) & 0x1;
        double y = ((double)mantissa) / 8.0;
        double result = LNNSMath::Exp(y);
        return (sign == 1) ? -result : result;
    }

    __host__ __device__
    int toInt() const {
        if(value == LOG_ZERO) return 0;
        if(value == LOG_NAN) return 0;
        int16_t mantissa = ((value & 0x4000)<<1) | (value & 0x7fff);
        uint16_t sign = (value >> 15) & 0x1;
        float y = ((float)mantissa) / 8.0;
        float result = LNNSMath::Exp(y);
        return (sign == 1) ? static_cast<int>(-result) : static_cast<int>(result);
    }

    __host__ __device__
    unsigned int toUInt() const {
        if(value == LOG_ZERO) return 0;
        if(value == LOG_NAN) return 0;
        int16_t mantissa = ((value & 0x4000)<<1) | (value & 0x7fff);
        float y = ((float)mantissa) / 8.0;
        float result = LNNSMath::Exp(y);
        return static_cast<unsigned int>(result);
    }


};
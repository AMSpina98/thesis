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

    template <typename T>
    __host__ __device__ inline void swap(T& a, T& b) {
    #ifdef __CUDA_ARCH__
        T temp = a;
        a = b;
        b = temp;
    #else
        std::swap(a, b);
    #endif
    }

}

template <typename T_, int PRECISION_>
struct LNNS { 
    using StorageType = T_; // x-bit storage uint type for LNNS representation
    using SignedStorageType = std::make_signed_t<StorageType>;
    static constexpr StorageType SIGN_BITS = 1; //ALWAYS 1
    static constexpr StorageType FRACTIONAL_BITS = PRECISION_; //Precision bits
    static constexpr StorageType INT_BITS = sizeof(StorageType) * 8 - SIGN_BITS - FRACTIONAL_BITS; //Range bits
    static constexpr StorageType MANTISSA_BITS = FRACTIONAL_BITS + INT_BITS;
    static constexpr StorageType MANTISSA_SIGN_BIT = MANTISSA_BITS - 1;
    static constexpr StorageType MANTISSA_SIGN_MASK = StorageType(1) << MANTISSA_SIGN_BIT;
    static constexpr float SCALE_FACTOR = static_cast<float>(StorageType(1) << FRACTIONAL_BITS);
    static constexpr StorageType SIGN_MASK = StorageType(1) << MANTISSA_BITS;
    static constexpr SignedStorageType  MAX_MANTISSA = (1 << (MANTISSA_BITS - 1)) - 1;
    static constexpr SignedStorageType  MIN_MANTISSA = -((1 << (MANTISSA_BITS - 1)) - 1);
    static constexpr StorageType MANTISSA_MASK = static_cast<StorageType>((StorageType(1) << MANTISSA_BITS) - 1);
    // Define reserved values
    static constexpr StorageType LOG_ZERO = MANTISSA_SIGN_MASK;
    static constexpr StorageType LOG_NAN = SIGN_MASK | MANTISSA_SIGN_MASK;

    static constexpr StorageType STORAGE_BITS = sizeof(StorageType) * 8;

    //static_assert(FRACTIONAL_BITS >= 1 && FRACTIONAL_BITS < STORAGE_BITS - SIGN_BITS, "PRECISION_ must leave room for at least 1 integer/range bit");


    StorageType value;

    public:
    __host__ __device__
    static LNNS raw(StorageType v) {
        return LNNS(v, RawTag{});
    }

    private:
    struct RawTag {};

    __host__ __device__
    LNNS(StorageType v, RawTag) : value(v) {}
    
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
        SignedStorageType mantissa = LNNSMath::NearbyInt(y * SCALE_FACTOR);
        if(mantissa > MAX_MANTISSA) mantissa = MAX_MANTISSA;
        if(mantissa < MIN_MANTISSA) mantissa = MIN_MANTISSA;
        StorageType z = static_cast<StorageType>(mantissa) & MANTISSA_MASK;
        value = (sign << MANTISSA_BITS) | z;
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
        SignedStorageType mantissa = LNNSMath::NearbyInt(y * SCALE_FACTOR);
        if(mantissa > MAX_MANTISSA) mantissa = MAX_MANTISSA;
        if(mantissa < MIN_MANTISSA) mantissa = MIN_MANTISSA;
        StorageType z = static_cast<StorageType>(mantissa) & MANTISSA_MASK;
        value = (sign << MANTISSA_BITS) | z;
    }

    __host__ __device__ LNNS(unsigned int x){
        if(x == 0.0){
            value = LOG_ZERO;
            return;
        }
        float y = LNNSMath::Log(LNNSMath::Abs(x));
        SignedStorageType mantissa = LNNSMath::NearbyInt(y * SCALE_FACTOR);
        if(mantissa > MAX_MANTISSA) mantissa = MAX_MANTISSA;
        if(mantissa < MIN_MANTISSA) mantissa = MIN_MANTISSA;
        value = static_cast<StorageType>(mantissa) & MANTISSA_MASK;
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
        SignedStorageType mantissa = LNNSMath::NearbyInt(y * SCALE_FACTOR);
        if(mantissa > MAX_MANTISSA) mantissa = MAX_MANTISSA;
        if(mantissa < MIN_MANTISSA) mantissa = MIN_MANTISSA;
        StorageType z = static_cast<StorageType>(mantissa) & MANTISSA_MASK;
        value = (sign << MANTISSA_BITS) | z;
    }


    LNNS() = default;
    __host__ __device__ explicit operator StorageType() const { return value; }
    

    __host__ __device__ LNNS operator+(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return LNNS::raw(static_cast<StorageType>(LOG_NAN));
        if(value == LOG_ZERO) return other;
        if(other.value == LOG_ZERO) return *this;
        SignedStorageType mantissaA = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        SignedStorageType mantissaB = ((other.value & MANTISSA_SIGN_MASK)<<1) | (other.value & MANTISSA_MASK);
        StorageType signA = value & SIGN_MASK;
        StorageType signB = other.value & SIGN_MASK;
        float s = (signA == signB) ? 1.0f : -1.0f; 
        if(mantissaA == mantissaB && s == -1.0f) return LNNS::raw(static_cast<StorageType>(LOG_ZERO));
        bool swapped = (mantissaB > mantissaA);
        SignedStorageType mMax = swapped ? mantissaB : mantissaA;
        SignedStorageType mMin = swapped ? mantissaA : mantissaB;
        StorageType signA_new  = swapped ? signB : signA;
        float ya = ((float)mMax) / SCALE_FACTOR;
        float diff = ((float)(mMin - mMax)) / SCALE_FACTOR;
        float yc = ya + LNNSMath::Log(1.0f + (s * LNNSMath::Exp(diff)));
        SignedStorageType mantissaC = (LNNSMath::NearbyInt(yc * SCALE_FACTOR));
        if(mantissaC > MAX_MANTISSA) mantissaC = MAX_MANTISSA;
        if(mantissaC < MIN_MANTISSA) mantissaC = MIN_MANTISSA;
        return LNNS::raw(static_cast<StorageType>(signA_new | (mantissaC & MANTISSA_MASK)));
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
        SignedStorageType mantissaA = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        SignedStorageType mantissaB = ((other.value & MANTISSA_SIGN_MASK)<<1) | (other.value & MANTISSA_MASK);
        StorageType signA = value & SIGN_MASK;
        StorageType signB = other.value & SIGN_MASK;
        float s = (signA == signB) ? 1.0f : -1.0f; 
        if(mantissaA == mantissaB && s == -1.0f){
            value = LOG_ZERO;
            return *this;
        }
        bool swapped = (mantissaB > mantissaA);
        SignedStorageType mMax = swapped ? mantissaB : mantissaA;
        SignedStorageType mMin = swapped ? mantissaA : mantissaB;
        StorageType signA_new  = swapped ? signB : signA;
        float ya = ((float)mMax) / SCALE_FACTOR;
        float diff = ((float)(mMin - mMax)) / SCALE_FACTOR;
        float yc = ya + LNNSMath::Log(1.0f + (s * LNNSMath::Exp(diff)));
        SignedStorageType mantissaC = (LNNSMath::NearbyInt(yc * SCALE_FACTOR));
        if(mantissaC > MAX_MANTISSA) mantissaC = MAX_MANTISSA;
        if(mantissaC < MIN_MANTISSA) mantissaC = MIN_MANTISSA;
        value = signA_new | (mantissaC & MANTISSA_MASK);
        return *this;
    }

    __host__ __device__ LNNS operator-(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return LNNS::raw(static_cast<StorageType>(LOG_NAN));
        if(other.value == LOG_ZERO) return *this;
        if(value == LOG_ZERO) return LNNS::raw(SIGN_MASK ^ other.value);
        SignedStorageType mantissaA = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        SignedStorageType mantissaB = ((other.value & MANTISSA_SIGN_MASK)<<1) | (other.value & MANTISSA_MASK);
        StorageType signA = value & SIGN_MASK;
        StorageType signB = (other.value & SIGN_MASK) ^ SIGN_MASK;
        float s = (signA == signB) ? 1.0f : -1.0f;
        if(mantissaA == mantissaB && s == -1.0f) return LNNS::raw(static_cast<StorageType>(LOG_ZERO));
        bool swapped = (mantissaB > mantissaA);
        SignedStorageType mMax = swapped ? mantissaB : mantissaA;
        SignedStorageType mMin = swapped ? mantissaA : mantissaB;
        StorageType signA_new  = swapped ? signB : signA;
        float ya = ((float)mMax) / SCALE_FACTOR;
        float diff = ((float)(mMin - mMax)) / SCALE_FACTOR;
        float yc = ya + LNNSMath::Log(1.0f + (s * LNNSMath::Exp(diff)));
        SignedStorageType mantissaC = (LNNSMath::NearbyInt(yc * SCALE_FACTOR));
        if(mantissaC > MAX_MANTISSA) mantissaC = MAX_MANTISSA;
        if(mantissaC < MIN_MANTISSA) mantissaC = MIN_MANTISSA;
        return LNNS::raw(static_cast<StorageType>(signA_new | (mantissaC & MANTISSA_MASK)));
    }

    __host__ __device__ LNNS& operator-=(const LNNS& other) {
        if(value == LOG_NAN || other.value == LOG_NAN) {
            value = LOG_NAN;
            return *this;
        }
        if(other.value == LOG_ZERO) return *this;
        if(value == LOG_ZERO) {
            value = SIGN_MASK ^ other.value;
            return *this;
        }
        SignedStorageType mantissaA = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        SignedStorageType mantissaB = ((other.value & MANTISSA_SIGN_MASK)<<1) | (other.value & MANTISSA_MASK);
        StorageType signA = value & SIGN_MASK;
        StorageType signB = (other.value & SIGN_MASK) ^ SIGN_MASK;
        float s = (signA == signB) ? 1.0f : -1.0f;
        if(mantissaA == mantissaB && s == -1.0f){
            value = LOG_ZERO;
            return *this;
        }
        bool swapped = (mantissaB > mantissaA);
        SignedStorageType mMax = swapped ? mantissaB : mantissaA;
        SignedStorageType mMin = swapped ? mantissaA : mantissaB;
        StorageType signA_new  = swapped ? signB : signA;
        float ya = ((float)mMax) / SCALE_FACTOR;
        float diff = ((float)(mMin - mMax)) / SCALE_FACTOR;
        float yc = ya + LNNSMath::Log(1.0f + (s * LNNSMath::Exp(diff)));
        SignedStorageType mantissaC = (LNNSMath::NearbyInt(yc * SCALE_FACTOR));
        if(mantissaC > MAX_MANTISSA) mantissaC = MAX_MANTISSA;
        if(mantissaC < MIN_MANTISSA) mantissaC = MIN_MANTISSA;
        value = signA_new | (mantissaC & MANTISSA_MASK);
        return *this;
    }

    __host__ __device__ LNNS operator*(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return LNNS::raw(static_cast<StorageType>(LOG_NAN));
        if(value == LOG_ZERO || other.value == LOG_ZERO) return LNNS::raw(static_cast<StorageType>(LOG_ZERO));
        SignedStorageType mantissaA = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        SignedStorageType mantissaB = ((other.value & MANTISSA_SIGN_MASK)<<1) | (other.value & MANTISSA_MASK);
        StorageType signA = value & SIGN_MASK;
        StorageType signB = other.value & SIGN_MASK;
        SignedStorageType mantissaC = mantissaA + mantissaB;
        if(mantissaC > MAX_MANTISSA) mantissaC = MAX_MANTISSA;
        if(mantissaC < MIN_MANTISSA) mantissaC = MIN_MANTISSA;
        return LNNS::raw(static_cast<StorageType>((signA ^ signB) | (mantissaC & MANTISSA_MASK)));
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
        SignedStorageType mantissaA = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        SignedStorageType mantissaB = ((other.value & MANTISSA_SIGN_MASK)<<1) | (other.value & MANTISSA_MASK);
        StorageType signA = value & SIGN_MASK;
        StorageType signB = other.value & SIGN_MASK;
        SignedStorageType mantissaC = mantissaA + mantissaB;
        if(mantissaC > MAX_MANTISSA) mantissaC = MAX_MANTISSA;
        if(mantissaC < MIN_MANTISSA) mantissaC = MIN_MANTISSA;
        value = (signA ^ signB) | (mantissaC & MANTISSA_MASK);
        return *this;
    }

    __host__ __device__ LNNS operator/(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN || other.value == LOG_ZERO) return LNNS::raw(static_cast<StorageType>(LOG_NAN));
        if(value == LOG_ZERO) return LNNS::raw(static_cast<StorageType>(LOG_ZERO));
        SignedStorageType mantissaA = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        SignedStorageType mantissaB = ((other.value & MANTISSA_SIGN_MASK)<<1) | (other.value & MANTISSA_MASK);
        StorageType signA = value & SIGN_MASK;
        StorageType signB = other.value & SIGN_MASK;
        SignedStorageType mantissaC = mantissaA - mantissaB;
        if(mantissaC > MAX_MANTISSA) mantissaC = MAX_MANTISSA;
        if(mantissaC < MIN_MANTISSA) mantissaC = MIN_MANTISSA;
        return LNNS::raw(static_cast<StorageType>((signA ^ signB) | (mantissaC & MANTISSA_MASK)));
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
        SignedStorageType mantissaA = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        SignedStorageType mantissaB = ((other.value & MANTISSA_SIGN_MASK)<<1) | (other.value & MANTISSA_MASK);
        StorageType signA = value & SIGN_MASK;
        StorageType signB = other.value & SIGN_MASK;
        SignedStorageType mantissaC = mantissaA - mantissaB;
        if(mantissaC > MAX_MANTISSA) mantissaC = MAX_MANTISSA;
        if(mantissaC < MIN_MANTISSA) mantissaC = MIN_MANTISSA;
        value = (signA ^ signB) | (mantissaC & MANTISSA_MASK);
        return *this;
    }

    __host__ __device__ bool operator<(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return false;
        SignedStorageType mantissaA = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        SignedStorageType mantissaB = ((other.value & MANTISSA_SIGN_MASK)<<1) | (other.value & MANTISSA_MASK);
        StorageType signA = value & SIGN_MASK;
        StorageType signB = other.value & SIGN_MASK;
        if(signA != signB) return signA;
        if(signA == 0) return mantissaA < mantissaB;
        else return mantissaA > mantissaB;
    }

    __host__ __device__ bool operator<=(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return false;
        SignedStorageType mantissaA = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        SignedStorageType mantissaB = ((other.value & MANTISSA_SIGN_MASK)<<1) | (other.value & MANTISSA_MASK);
        StorageType signA = value & SIGN_MASK;
        StorageType signB = other.value & SIGN_MASK;
        if(signA != signB) return signA;
        if(signA == 0) return mantissaA <= mantissaB;
        else return mantissaA >= mantissaB;
    }

    __host__ __device__ bool operator>(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return false;
        SignedStorageType mantissaA = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        SignedStorageType mantissaB = ((other.value & MANTISSA_SIGN_MASK)<<1) | (other.value & MANTISSA_MASK);
        StorageType signA = value & SIGN_MASK;
        StorageType signB = other.value & SIGN_MASK;
        if(signA != signB) return signB;
        if(signA == 0) return mantissaA > mantissaB;
        else return mantissaA < mantissaB;
    }

    __host__ __device__ bool operator>=(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return false;
        SignedStorageType mantissaA = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        SignedStorageType mantissaB = ((other.value & MANTISSA_SIGN_MASK)<<1) | (other.value & MANTISSA_MASK);
        StorageType signA = value & SIGN_MASK;
        StorageType signB = other.value & SIGN_MASK;
        if(signA != signB) return signB;
        if(signA == 0) return mantissaA >= mantissaB;
        else return mantissaA <= mantissaB;
    }

    __host__ __device__ bool operator==(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return false;
        SignedStorageType mantissaA = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        SignedStorageType mantissaB = ((other.value & MANTISSA_SIGN_MASK)<<1) | (other.value & MANTISSA_MASK);
        StorageType signA = value & SIGN_MASK;
        StorageType signB = other.value & SIGN_MASK;
        if(signA != signB) return false;
        return mantissaA == mantissaB;
    }

    __host__ __device__ bool operator!=(const LNNS& other) const {
        if(value == LOG_NAN || other.value == LOG_NAN) return true;
        SignedStorageType mantissaA = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        SignedStorageType mantissaB = ((other.value & MANTISSA_SIGN_MASK)<<1) | (other.value & MANTISSA_MASK);
        StorageType signA = value & SIGN_MASK;
        StorageType signB = other.value & SIGN_MASK;
        if(signA != signB) return true;
        return mantissaA != mantissaB;
    }


    __host__ __device__
    float toFloat() const {
        if(value == LOG_ZERO) return 0.0f;
        if(value == LOG_NAN) return LNNSMath::Quiet_NAN();
        SignedStorageType mantissa = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
	    StorageType sign = value & SIGN_MASK;
	    float y = ((float)mantissa) / SCALE_FACTOR;
	    float result = LNNSMath::Exp(y);
        return (sign != 0) ? -result : result;
    }

    __host__ __device__
    double toDouble() const {
        if(value == LOG_ZERO) return 0.0;
        if(value == LOG_NAN) return LNNSMath::Quiet_NAN();
        SignedStorageType mantissa = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        StorageType sign = value & SIGN_MASK;
        double y = ((double)mantissa) / SCALE_FACTOR;
        double result = LNNSMath::Exp(y);
        return (sign !=0) ? -result : result;
    }

    __host__ __device__
    int toInt() const {
        if(value == LOG_ZERO) return 0;
        if(value == LOG_NAN) return 0;
        SignedStorageType mantissa = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        StorageType sign = value & SIGN_MASK;
        float y = ((float)mantissa) / SCALE_FACTOR;
        float result = LNNSMath::Exp(y);
        return (sign != 0) ? static_cast<int>(-result) : static_cast<int>(result);
    }

    __host__ __device__
    unsigned int toUInt() const {
        if(value == LOG_ZERO) return 0;
        if(value == LOG_NAN) return 0;
        SignedStorageType mantissa = ((value & MANTISSA_SIGN_MASK)<<1) | (value & MANTISSA_MASK);
        float y = ((float)mantissa) / SCALE_FACTOR;
        float result = LNNSMath::Exp(y);
        return static_cast<unsigned int>(result);
    }

    template <typename U = StorageType, typename = typename std::enable_if<!std::is_same<U, float>::value>::type>
    __host__ __device__ operator float() const { return toFloat(); }

    template <typename U = StorageType, typename = typename std::enable_if<!std::is_same<U, double>::value>::type>
    __host__ __device__ operator double() const { return toDouble(); }

    template <typename U = StorageType, typename = typename std::enable_if<!std::is_same<U, int>::value>::type>
    __host__ __device__ operator int() const { return toInt(); }

    template <typename U = StorageType, typename = typename std::enable_if<!std::is_same<U, unsigned int>::value>::type>
    __host__ __device__ operator unsigned int() const { return toUInt(); }
};
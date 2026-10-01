#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>
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

namespace EncapsulatorMath {

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
        return static_cast<int>(std::nearbyintf(x));
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
    static_assert(std::is_floating_point<T>::value,
        "Encapsulator<T> e' pensato per sostituire float/double (Strategia 1 -- "
        "forzare add/sub ordinarie sulle SFU). Con T intero, static_cast<T>(x) "
        "su un risultato negativo e T unsigned e' undefined behavior, e non c'e' "
        "una semantica di arrotondamento condivisa col resto del codice: fuori "
        "scope per questa sperimentazione, deliberatamente non supportato.");

    T value;

    __host__ __device__
    Encapsulator() = default;

    __host__ __device__
    Encapsulator(T v) : value(v) {}

    __host__ __device__
    operator T() const { return value; }

    template <typename U = T, typename = typename std::enable_if<!std::is_same<U, float>::value>::type>
    __host__ __device__ Encapsulator(float x) {
        value = static_cast<T>(x);
    }
    template <typename U = T, typename = typename std::enable_if<!std::is_same<U, float>::value>::type>
    __host__ __device__ Encapsulator(double x) {
        value = static_cast<T>(x);
    }
    template <typename U = T, typename = typename std::enable_if<!std::is_same<U, float>::value>::type>
    __host__ __device__ Encapsulator(unsigned int x) {
        value = static_cast<T>(x);
    }
    template <typename U = T, typename = typename std::enable_if<!std::is_same<U, float>::value>::type>
    __host__ __device__ Encapsulator(int x) {
        value = static_cast<T>(x);
    }

    __host__ __device__ Encapsulator operator+(const Encapsulator& other) const {
        float a = toFloat();
        float b = other.toFloat();
        if (b == 0.0f) return *this;
        if (a == 0.0f) return other;

        float signA = (a < 0.0f) ? -1.0f : 1.0f;
        float signB = (b < 0.0f) ? -1.0f : 1.0f;

        a = EncapsulatorMath::Log(EncapsulatorMath::Abs(a));
        b = EncapsulatorMath::Log(EncapsulatorMath::Abs(b));

        float s = (signA == signB) ? 1.0f : -1.0f;
        if (a == b && s == -1.0f) {
            return Encapsulator<T>(0.0f);
        }

        bool swapped = (b > a);
        float signC = swapped ? signB : signA;
        float aMax  = swapped ? b : a;
        float aMin  = swapped ? a : b;

        float ya = aMax + EncapsulatorMath::Log(1.0f + (s * EncapsulatorMath::Exp(aMin - aMax)));
        ya = signC * EncapsulatorMath::Exp(ya);

        return Encapsulator<T>(ya);
    }

    __host__ __device__ Encapsulator& operator+=(const Encapsulator& other) {
        float a = toFloat();
        float b = other.toFloat();
        if (b == 0.0f) {
            return *this;
        }
        if (a == 0.0f) {
            value = other.value;
            return *this;
        }

        float signA = (a < 0.0f) ? -1.0f : 1.0f;
        float signB = (b < 0.0f) ? -1.0f : 1.0f;

        a = EncapsulatorMath::Log(EncapsulatorMath::Abs(a));
        b = EncapsulatorMath::Log(EncapsulatorMath::Abs(b));

        float s = (signA == signB) ? 1.0f : -1.0f;
        if (a == b && s == -1.0f) {
            // FIX: prima si ritornava un Encapsulator per valore da una
            // funzione che deve ritornare un riferimento -- non compilava.
            value = static_cast<T>(0.0f);
            return *this;
        }

        bool swapped = (b > a);
        float signC = swapped ? signB : signA;
        float aMax  = swapped ? b : a;
        float aMin  = swapped ? a : b;

        float ya = aMax + EncapsulatorMath::Log(1.0f + (s * EncapsulatorMath::Exp(aMin - aMax)));
        ya = signC * EncapsulatorMath::Exp(ya);

        value = static_cast<T>(ya);
        return *this;
    }

    // ------------------------------------------------------------------
    // Sottrazione: a-b = a+(-b). Il segno di "b" viene estratto GIA'
    // negato (signB = sign(-b)), esattamente come nel fix fatto per
    // LNNS_type.cuh: cosi' il corpo del calcolo e' identico a operator+.
    // ------------------------------------------------------------------
    __host__ __device__ Encapsulator operator-(const Encapsulator& other) const {
        float a = toFloat();
        float b = other.toFloat();
        if (b == 0.0f) return *this;
        if (a == 0.0f) {
            // FIX: 0 - b = -b, non "b" (bug di segno copiato da operator+)
            return Encapsulator<T>(-b);
        }

        float signA = (a < 0.0f) ? -1.0f : 1.0f;
        float signB = (b > 0.0f) ? -1.0f : 1.0f;   // sign(-b), gia' negato

        a = EncapsulatorMath::Log(EncapsulatorMath::Abs(a));
        b = EncapsulatorMath::Log(EncapsulatorMath::Abs(b));

        float s = (signA == signB) ? 1.0f : -1.0f;
        if (a == b && s == -1.0f) {
            return Encapsulator<T>(0.0f);
        }

        bool swapped = (b > a);
        float signC = swapped ? signB : signA;
        float aMax  = swapped ? b : a;
        float aMin  = swapped ? a : b;

        float ya = aMax + EncapsulatorMath::Log(1.0f + (s * EncapsulatorMath::Exp(aMin - aMax)));
        ya = signC * EncapsulatorMath::Exp(ya);

        return Encapsulator<T>(ya);
    }

    __host__ __device__ Encapsulator& operator-=(const Encapsulator& other) {
        float a = toFloat();
        float b = other.toFloat();
        if (b == 0.0f) {
            return *this;
        }
        if (a == 0.0f) {
            // FIX: stesso bug di segno di operator-, qui sulla forma compound
            value = static_cast<T>(-b);
            return *this;
        }

        float signA = (a < 0.0f) ? -1.0f : 1.0f;
        float signB = (b > 0.0f) ? -1.0f : 1.0f;

        a = EncapsulatorMath::Log(EncapsulatorMath::Abs(a));
        b = EncapsulatorMath::Log(EncapsulatorMath::Abs(b));

        float s = (signA == signB) ? 1.0f : -1.0f;
        if (a == b && s == -1.0f) {
            // FIX: stesso problema di ritorno-per-valore di operator+=
            value = static_cast<T>(0.0f);
            return *this;
        }

        bool swapped = (b > a);
        float signC = swapped ? signB : signA;
        float aMax  = swapped ? b : a;
        float aMin  = swapped ? a : b;

        float ya = aMax + EncapsulatorMath::Log(1.0f + (s * EncapsulatorMath::Exp(aMin - aMax)));
        ya = signC * EncapsulatorMath::Exp(ya);

        value = static_cast<T>(ya);
        return *this;
    }

    __host__ __device__ Encapsulator operator*(const Encapsulator& other) const {
        float a = toFloat();
        float b = other.toFloat();
        bool sign = (a < 0.0f) ^ (b < 0.0f);
        float ma, mb;
        int ea, eb;
        ma = frexpf(EncapsulatorMath::Abs(a), &ea);
        mb = frexpf(EncapsulatorMath::Abs(b), &eb);
        float result = EncapsulatorMath::Exp(EncapsulatorMath::Log(EncapsulatorMath::Abs(ma)) +
                                              EncapsulatorMath::Log(EncapsulatorMath::Abs(mb)));
        result = ldexpf(result, (ea + eb));
        if (sign) result = -result;
        return Encapsulator<T>(result);
    }

    __host__ __device__ Encapsulator& operator*=(const Encapsulator& other) {
        float a = toFloat();
        float b = other.toFloat();
        bool sign = (a < 0.0f) ^ (b < 0.0f);
        float ma, mb;
        int ea, eb;
        ma = frexpf(EncapsulatorMath::Abs(a), &ea);
        mb = frexpf(EncapsulatorMath::Abs(b), &eb);
        float result = EncapsulatorMath::Exp(EncapsulatorMath::Log(EncapsulatorMath::Abs(ma)) +
                                              EncapsulatorMath::Log(EncapsulatorMath::Abs(mb)));
        result = ldexpf(result, (ea + eb));
        if (sign) result = -result;
        value = static_cast<T>(result);
        return *this;
    }

    __host__ __device__ Encapsulator operator/(const Encapsulator& other) const {
        float a = toFloat();
        float b = other.toFloat();
        bool sign = (a < 0.0f) ^ (b < 0.0f);
        float ma, mb;
        int ea, eb;
        ma = frexpf(EncapsulatorMath::Abs(a), &ea);
        mb = frexpf(EncapsulatorMath::Abs(b), &eb);
        float result = EncapsulatorMath::Exp(EncapsulatorMath::Log(EncapsulatorMath::Abs(ma)) -
                                              EncapsulatorMath::Log(EncapsulatorMath::Abs(mb)));
        result = ldexpf(result, (ea - eb));
        if (sign) result = -result;
        return Encapsulator<T>(result);
    }

    __host__ __device__ Encapsulator& operator/=(const Encapsulator& other) {
        float a = toFloat();
        float b = other.toFloat();
        bool sign = (a < 0.0f) ^ (b < 0.0f);
        float ma, mb;
        int ea, eb;
        ma = frexpf(EncapsulatorMath::Abs(a), &ea);
        mb = frexpf(EncapsulatorMath::Abs(b), &eb);
        float result = EncapsulatorMath::Exp(EncapsulatorMath::Log(EncapsulatorMath::Abs(ma)) -
                                              EncapsulatorMath::Log(EncapsulatorMath::Abs(mb)));
        result = ldexpf(result, (ea - eb));
        if (sign) result = -result;
        value = static_cast<T>(result);
        return *this;
    }

    __host__ __device__ bool operator<(const Encapsulator& other) const {
        return toFloat() < other.toFloat();
    }
    __host__ __device__ bool operator<=(const Encapsulator& other) const {
        return toFloat() <= other.toFloat();
    }
    __host__ __device__ bool operator>(const Encapsulator& other) const {
        return toFloat() > other.toFloat();
    }
    __host__ __device__ bool operator>=(const Encapsulator& other) const {
        return toFloat() >= other.toFloat();
    }
    __host__ __device__ bool operator==(const Encapsulator& other) const {
        return toFloat() == other.toFloat();
    }
    __host__ __device__ bool operator!=(const Encapsulator& other) const {
        return toFloat() != other.toFloat();
    }

    __host__ __device__ float toFloat() const {
        return static_cast<float>(value);
    }
    __host__ __device__ double toDouble() const {
        return static_cast<double>(value);
    }
    __host__ __device__ int toInt() const {
        return static_cast<int>(value);
    }
    __host__ __device__ unsigned int toUInt() const {
        return static_cast<unsigned int>(value);
    }
    
    template <typename U = T, typename = typename std::enable_if<!std::is_same<U, float>::value>::type>
    __host__ __device__ operator float() const { return toFloat(); }

    template <typename U = T, typename = typename std::enable_if<!std::is_same<U, double>::value>::type>
    __host__ __device__ operator double() const { return toDouble(); }

    template <typename U = T, typename = typename std::enable_if<!std::is_same<U, int>::value>::type>
    __host__ __device__ operator int() const { return toInt(); }

    template <typename U = T, typename = typename std::enable_if<!std::is_same<U, unsigned int>::value>::type>
    __host__ __device__ operator unsigned int() const { return toUInt(); }
};

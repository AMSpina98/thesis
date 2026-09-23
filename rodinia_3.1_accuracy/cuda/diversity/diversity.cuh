#pragma once

#ifdef __cplusplus
extern "C++" {
#endif


#include "converters.cuh"




template <typename T, typename V>
__device__ T dMul(T A, V B) {
    //return A * B;
    float a = Converter<T>::to(A);
    float b = Converter<V>::to(B);
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


template <typename T, typename V>
__device__ T dDiv(T A, V B) {
    //return A / B;
    float a = Converter<T>::to(A);
    float b = Converter<V>::to(B);
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

template <typename T, typename V>
__device__ T dAdd(T A, V B) {
    //return A + B;
    float a = Converter<T>::to(A);
    float b = Converter<V>::to(B);
    float c = max(fabs(a), fabs(b));
    if(c < 1.0f) c = 1.0f;
    a = dDiv(a,c);
    b = dDiv(b,c);
    float result = __logf(__expf(a) * __expf(b));
    result = dMul(result, c);
    return Converter<T>::from(result);
}

template <typename T, typename V>
__device__ T dSub(T A, V B) {
    //return A - B;
    float a = Converter<T>::to(A);
    float b = Converter<V>::to(B);
    float c = max(fabs(a), fabs(b));
    if(c < 1.0f) c = 1.0f;
    a = dDiv(a,c);
    b = dDiv(b,c);
    float result = __logf(__expf(a) / __expf(b));
    result = dMul(result, c);
    return Converter<T>::from(result);
}


#ifdef __cplusplus
}
#endif
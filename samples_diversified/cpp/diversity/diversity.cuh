#pragma once

#ifdef __cplusplus
extern "C++" {
#endif


#include "converters.cuh"




template <typename T, typename V>
__device__ T dMul(T A, V B) {

    float a = Converter<T>::to(A);
    float b = Converter<V>::to(B);

    bool sign = (a < 0.0f) ^ (b < 0.0f);
    a = fabsf(a);
    b = fabsf(b);
    float result = __expf(__logf(a) + __logf(b));
    if (sign) result = -result;
    return Converter<T>::from(result);

}


template <typename T, typename V>
__device__ T dDiv(T A, V B) {

    float a = Converter<T>::to(A);
    float b = Converter<V>::to(B);

    bool sign = (a < 0.0f) ^ (b < 0.0f);
    a = fabsf(a);
    b = fabsf(b);
    float result = __expf(__logf(a) - __logf(b));
    if (sign) result = -result;
    return Converter<T>::from(result);
}

template <typename T, typename V>
__device__ T dAdd(T A, V B) {
    float a = Converter<T>::to(A);
    float b = Converter<V>::to(B);

    float c = max(a,b);
    a = dDiv(a,c);
    b = dDiv(b,c);
    float result = __logf(__expf(a) * __expf(b));
    result = dMul(result, c);
    return Converter<T>::from(result);
}


#ifdef __cplusplus
}
#endif
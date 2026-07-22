#pragma once

#ifdef __cplusplus
extern "C++" {
#endif


#include "converters.cuh"
#include <assert.h>


template <typename T, typename V>
__device__ T dMul(T A, V B) {
    float a = Converter<T>::to(A);
    float b = Converter<V>::to(B);
	float result = a*b;/*
    if (a == 0.0f || b == 0.0f) return Converter<T>::from(0.0f);
    bool sign = (a < 0.0f) ^ (b < 0.0f);
    a = fabsf(a); b = fabsf(b);
    float result = __expf(__logf(a) + __logf(b));
    if (sign) result = -result;*/
    return Converter<T>::from(result);
}

template <typename T, typename V>
__device__ T dDiv(T A, V B) {
    float a = Converter<T>::to(A);
    float b = Converter<V>::to(B);
	float result = a/b;/*
    if (a == 0.0f && b != 0.0f) return Converter<T>::from(0.0f);
    bool sign = (a < 0.0f) ^ (b < 0.0f);
    a = fabsf(a); b = fabsf(b);
    float result = __expf(__logf(a) - __logf(b));
    if (sign) result = -result;*/
    return Converter<T>::from(result);
}


template <typename T, typename V>
__device__ T dAdd(T A, V B) {
    float a = Converter<T>::to(A);
    float b = Converter<V>::to(B);
	float res = a + b;/*
	float c = max(fabs(a), fabs(b));
	c = max(c, 2.0f);
	a = dDiv(a, c);
	b = dDiv(b, c);
	float res = __logf(__expf(a)*__expf(b));
	res = dMul(res, c);
*/

    return Converter<T>::from(res);
}

template <typename T, typename V>
__device__ T dSub(T A, V B) {
    float a = Converter<T>::to(A);
    float b = Converter<V>::to(B);
	float c = max(fabs(a), fabs(b));
	c = max(c, 2.0f);
	a = dDiv(a, c);
	b = dDiv(b, c);
	float res = __logf(__expf(a)/__expf(b));
	res = dMul(res, c);
    return Converter<T>::from(res);
}



#ifdef __cplusplus
}
#endif

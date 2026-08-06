#pragma once

#include <cstdint>



//Converter
template<typename T>
struct Converter{
	__host__ __device__ static float to(T x);
	__host__ __device__ static T from(float x);
};


template<>
struct Converter<int> {
    __host__ __device__ static float to(int x){
        return (float)x;
    }

    __host__ __device__ static int from(float x){
#ifdef __CUDA_ARCH__
        return __float2int_rn(x);
#else
        return static_cast<int>(x);
#endif
    }
};

template<>
struct Converter<double> {
    __host__ __device__ static float to(double x) {
        return static_cast<float>(x);
    }

    __host__ __device__ static double from(float x) {
        return static_cast<double>(x);
    }
};

template<>
struct Converter<float> {
	__host__ __device__ static float to(float x){
		return x;	
	}
	__host__ __device__ static float from(float x){
		return x;
	}
};


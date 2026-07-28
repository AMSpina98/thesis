#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#ifdef __CUDA_BF16_H__
#include <cuda_bf16.h>
#endif
#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#define HAS_FP8 1
#else
#define HAS_FP8 0
#endif


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
struct Converter<unsigned int> {
    __host__ __device__ static float to(unsigned int x){
        return (float)x;
    }

    __host__ __device__ static unsigned int from(float x){
#ifdef __CUDA_ARCH__
        return __float2uint_rn(x);
#else
        return static_cast<unsigned int>(x);
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

template<>
struct Converter<__half> {
	__host__ __device__ static float to(__half x){
		return __half2float(x);	
	}
	__host__ __device__ static __half from(float x){
		return __float2half(x);
	}
};



#ifdef __CUDA_BF16_H__

	template<>
	struct Converter<__nv_bfloat16> {
		__host__ __device__ static float to(__nv_bfloat16 x){
			return __bfloat162float(x);	
		}
		__host__ __device__ static __nv_bfloat16 from(float x){
			return __float2bfloat16(x);
		}
	};
#endif
#if defined(HAS_FP8) && HAS_FP8

	template<>
	struct Converter<__nv_fp8_e4m3> {
		__host__ __device__ static float to(__nv_fp8_e4m3 x){
			return static_cast<float>(x);
		}
		__host__ __device__ static __nv_fp8_e4m3 from(float x){
			return __nv_fp8_e4m3(x);
		}
	};

	template<>
	struct Converter<__nv_fp8_e5m2> {
		__host__ __device__ static float to(__nv_fp8_e5m2 x){
			return static_cast<float>(x);
		}
		__host__ __device__ static __nv_fp8_e5m2 from(float x){
			return __nv_fp8_e5m2(x);
		}
	};
#endif
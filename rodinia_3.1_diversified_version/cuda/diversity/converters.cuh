#pragma once

#ifdef __cplusplus
extern "C++" {
#endif


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
	__host__ __device__ static float to(int x) {
		return (float)x;
	}

	__host__ __device__ static int from(float x){
#ifdef __CUDA_ARCH__
		return __float2int_rn(x);
#else
		return lroundf(x);
#endif
	}
};

template<>
struct Converter<uint32_t> {
	__host__ __device__ static float to(uint32_t x) {
		return (float)x;
	}

	__host__ __device__ static uint32_t from(float x){
#ifdef __CUDA_ARCH__
		return (uint32_t)__float2int_rn(x);
#else
		return (uint32_t)lroundf(x);
#endif
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

template<>
struct Converter<double> {
	__host__ __device__ static float to(double x){
		return static_cast<float>(x);	
	}
	__host__ __device__ static double from(float x){
		return static_cast<double>(x);
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
	struct Converter<__NV_E4M3> {
		__host__ __device__ static float to(__NV_E4M3 x){
			return __nv_cvt_fp8_to_float(x, __NV_E4M3);	
		}
		__host__ __device__ static __NV_E4M3 from(float x){
			return __nv_cvt_float_to_fp8_rn(x, __NV_E4M3);
		}
	};

	template<>
	struct Converter<__NV_E5M2> {
		__host__ __device__ static float to(__NV_E5M2 x){
			return __nv_cvt_fp8_to_float(x, __NV_E5M2);	
		}
		__host__ __device__ static __NV_E5M2 from(float x){
			return __nv_cvt_float_to_fp8_rn(x, __NV_E5M2);
		}
	};
#endif

#ifdef __cplusplus
}
#endif


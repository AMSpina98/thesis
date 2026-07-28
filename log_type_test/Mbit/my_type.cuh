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

namespace NLFormat {
    constexpr int SIGN_BITS = 1;
    constexpr int BASE_BITS = 4;
    constexpr int MANTISSA_LENGTH = 27;

    constexpr double LN2 = 0.6931471805599453;
    constexpr double MANTISSA_FACTOR = static_cast<double>(MANTISSA_LENGTH) * LN2;
    constexpr uint32_t MANTISSA_SCALE = 1u << MANTISSA_LENGTH;

    constexpr uint32_t MAX_MANTISSA = (1u << MANTISSA_LENGTH) - 1;

    __device__ __constant__ float inv_log_base[16] = {
        // f(B) = 1/ln(B), B is the base of the logarithm, f(B) is the inverse of the natural logarithm of B:
        // it's used to convert natural logarithm to logarithm with base B and viceversa.
        50.49834979,  // 1.02
        20.49593431,  // 1.05
        12.99358721,  // 1.08
        10.49205869,  // 1.10
        7.17967697,   // 1.15
        5.48481495,   // 1.20
        3.81755393,   // 1.30
        2.99591172,   // 1.40
        2.46630346,   // 1.50
        2.12764315,   // 1.60
        1.88416939,   // 1.70
        1.70054688,   // 1.80
        1.55730501,   // 1.90
        1.44269504,   // 2.00
        1.00000000,    // e
        0.0 //to find
    };

    __device__ __constant__ float log_base[16] = {
        0.019802627,   // 1.02
        0.048790164,   // 1.05
        0.076961041,   // 1.08
        0.095310180,   // 1.10
        0.139761942,   // 1.15
        0.182321557,   // 1.20
        0.262364264,   // 1.30
        0.336472237,   // 1.40
        0.405465108,   // 1.50
        0.470003629,   // 1.60
        0.530628251,   // 1.70
        0.587786665,   // 1.80
        0.641853886,   // 1.90
        0.693147181,   // 2.00
        1.000000000,   // e
        0.0
    };

    __device__ __constant__ float base_changers[16][16] = {
        {1, 0.405873348, 0.257307165, 0.207770328, 0.141688266, 0.108613746, 0.075477609, 0.058853674, 0.048839288, 0.042132924, 0.03731921, 0.033690161, 0.030852236, 0.028569152, 0.019802627, 0},
        {2.463822777, 1, 0.633959253, 0.511909266, 0.349094777, 0.267605022, 0.185963451, 0.145005022, 0.120331351, 0.103808058, 0.09194792, 0.083006586, 0.076014441, 0.070389328, 0.048790164, 0},
        {3.886405576, 1.577388444, 1, 0.80747976, 0.550658068, 0.422117069, 0.293336599, 0.228729246, 0.189809282, 0.163745632, 0.145037587, 0.130933629, 0.1199043, 0.111031312, 0.076961041,0},
        {4.813006798, 1.953471185, 1.238421134, 1, 0.681946588, 0.522758699, 0.363274244, 0.283263133, 0.235063826, 0.202786051, 0.179617613, 0.162150973, 0.14849202, 0.137503524, 0.09531018, 0},
        {7.057747454, 2.864551591, 1.816008987, 1.466390502, 1, 0.766568391, 0.532701901, 0.415374367, 0.344695362, 0.297363539, 0.263389562, 0.237776647, 0.217747287, 0.201633861, 0.139761942, 0},
        {9.20693775, 3.736850652, 2.369011049, 1.912928474, 1.304515047, 1, 0.694917645, 0.541862112, 0.449660287, 0.387915211, 0.343595646, 0.310183214, 0.284054612, 0.263034406, 0.182321557, 0},
        {13.2489624, 5.377400731, 3.409052952, 2.75274126, 1.877222511, 1.43901944, 1, 0.779750113, 0.647069894, 0.558217529, 0.494440814, 0.446359675, 0.408760109, 0.378511623, 0.262364264, 0},
        {16.9912927, 6.89631286, 4.371981351, 3.530286453, 2.407466803, 1.845487953, 1.282462142, 1, 0.829842642, 0.715892848, 0.634101626, 0.572439384, 0.524219365, 0.485426827, 0.336472237, 0},
        {20.47531886, 8.310386223, 5.268446244, 4.25416371, 2.901112429, 2.223901086, 1.545428105, 1.205047739, 1, 0.862685058, 0.764122731, 0.689816786, 0.631709361, 0.584962501, 0.405465108, 0},
        {23.73440767, 9.633163512, 6.107033147, 4.931305661, 3.362887073, 2.577882931, 1.791416336, 1.39685709, 1.159171578, 1, 0.885749351, 0.799616013, 0.732259537, 0.678071905, 0.470003629, 0},
        {26.79585103, 10.87572178, 6.894764458, 5.567382751, 3.796657674, 2.910397763, 2.02248676, 1.577034279, 1.308690293, 1.128987561, 1, 0.902756532, 0.826711908, 0.765534746, 0.530628251, 0},
        {29.68225661, 12.04723687, 7.637457293, 6.167092184, 4.205627476, 3.223901086, 2.24034575, 1.746909851, 1.449660287, 1.250600268, 1.107718377, 1, 0.915763973, 0.847996907, 0.587786665, 0},
        {32.41256206, 13.15539509, 8.339984448, 6.734368642, 4.592479721, 3.520449789, 2.446422677, 1.907598358, 1.583006462, 1.365636021, 1.20961122, 1.09198443, 1, 0.925999419, 0.641853886, 0},
        {35.00278878, 14.20669908, 9.006468342, 7.272540897, 4.959484455, 3.801784017, 2.641926796, 2.060042717, 1.709511291, 1.474769847, 1.306276436, 1.179249585, 1.079914285, 1, 0.693147181, 0},
        {50.49834979, 20.49593431, 12.99358721, 10.49205869, 7.155023628, 5.484814948, 3.811494687, 2.972013412, 2.466303462, 2.127643145, 1.884558536, 1.701297528, 1.557986984, 1.442695041, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };
}


//Converter
template<typename T>
struct Converter{
	__device__ static NL to(T x, int base);
	__device__ static T from(NL x);
};


template<>
struct Converter<int> {
    __device__ static NL to(int x, int base){
        uint32_t sign = (x < 0);
        float ax = fabs(x);
        float I = __expf(-ax * NLFormat::log_base[base]);
        uint32_t mantissa = __float2uint_rn(I * NLFormat::MANTISSA_SCALE);
        if (mantissa == 0) mantissa = 1;
        if (mantissa > NLFormat::MAX_MANTISSA) mantissa = NLFormat::MAX_MANTISSA;
        return (sign << 31) | (base << NLFormat::MANTISSA_LENGTH) | mantissa;
    }

    __device__ static int from(NL x){
        uint32_t sign = (x >> 31) & 0x1;
        uint32_t base = (x >> NLFormat::MANTISSA_LENGTH) & 0xF;
        int32_t mantissa = x & ((1u << NLFormat::MANTISSA_LENGTH) - 1);
        float value = (NLFormat::MANTISSA_FACTOR - __logf(mantissa)) * NLFormat::inv_log_base[base];
        if (sign) value = -value;
        return __float2int_rn(value);
    }
};


template<>
struct Converter<unsigned int> {
    __device__ static NL to(unsigned int x, int base){
        float ax = (float)x;
        float I = __expf(-ax * NLFormat::log_base[base]);
        uint32_t mantissa = __float2uint_rn(I * NLFormat::MANTISSA_SCALE);
        mantissa = max(1u, mantissa);
        mantissa = min(NLFormat::MAX_MANTISSA, mantissa);
        return (base << NLFormat::MANTISSA_LENGTH) | mantissa;
    }


    __device__ static unsigned int from(NL x){
        uint32_t base = (x >> NLFormat::MANTISSA_LENGTH) & 0xF;
        uint32_t mantissa = x & NLFormat::MAX_MANTISSA;
        float value = (NLFormat::MANTISSA_FACTOR - __logf((float)mantissa)) * NLFormat::inv_log_base[base];
        return __float2uint_rn(value);
    }
};

template<>
struct Converter<double> {
    __device__ static NL to(double x, int base){
        uint32_t sign = (x < 0.0);
        double ax = fabs(x);
        double I = __expf(-ax * NLFormat::log_base[base]);
        uint32_t mantissa = (uint32_t) llrint(I * NLFormat::MANTISSA_SCALE);
        if (mantissa == 0) mantissa = 1;
        if (mantissa > NLFormat::MAX_MANTISSA) mantissa = NLFormat::MAX_MANTISSA;
        return (sign << 31) | (base << NLFormat::MANTISSA_LENGTH) | mantissa;
    }

    __device__ static double from(NL x) {
        uint32_t sign = (x >> 31) & 0x1;
        uint32_t base = (x >> NLFormat::MANTISSA_LENGTH) & 0xF;
        int32_t mantissa = x & ((1u << NLFormat::MANTISSA_LENGTH) - 1);
        double value = (NLFormat::MANTISSA_FACTOR - __logf(mantissa)) * NLFormat::inv_log_base[base];
        if (sign) value = -value;
        return value;
    }
};

template<>
struct Converter<float> {
	__device__ static NL to(float x, int base){
        uint32_t sign = (x < 0.0);
        float ax = fabs(x);
        float I = __expf(-ax * NLFormat::log_base[base]);
        uint32_t mantissa = __float2uint_rn(I * NLFormat::MANTISSA_SCALE);
        if (mantissa == 0) mantissa = 1;
        if (mantissa > NLFormat::MAX_MANTISSA) mantissa = NLFormat::MAX_MANTISSA;
        return (sign << 31) | (base << NLFormat::MANTISSA_LENGTH) | mantissa;
	}
	__device__ static float from(NL x){
        uint32_t sign = (x >> 31) & 0x1;
        uint32_t base = (x >> NLFormat::MANTISSA_LENGTH) & 0xF;
        int32_t mantissa = x & ((1u << NLFormat::MANTISSA_LENGTH) - 1);
        float value = (NLFormat::MANTISSA_FACTOR - __logf(mantissa)) * NLFormat::inv_log_base[base];
        if (sign) value = -value;
        return value;
	}
};
/*  other types to be added here, e.g. half, bfloat16, fp8, etc.
template<>
struct Converter<__half> {
	__device__ static NL to(__half x){
		return static_cast<NL>(x);
	}
	__device__ static __half from(NL x){
		return __float2half(static_cast<float>(x));
	}
};



#ifdef __CUDA_BF16_H__

	template<>
	struct Converter<__nv_bfloat16> {
		__device__ static NL to(__nv_bfloat16 x){
			return static_cast<NL>(x);	
		}
		__device__ static __nv_bfloat16 from(NL x){
			return __float2bfloat16(static_cast<float>(x));
		}
	};
#endif
#if defined(HAS_FP8) && HAS_FP8

	template<>
	struct Converter<__nv_fp8_e4m3> {
		__device__ static NL to(__nv_fp8_e4m3 x){
			return static_cast<NL>(x);
		}
		__device__ static __nv_fp8_e4m3 from(NL x){
			return __nv_fp8_e4m3(static_cast<float>(x));
		}
	};

	template<>
	struct Converter<__nv_fp8_e5m2> {
		__device__ static NL to(__nv_fp8_e5m2 x){
			return static_cast<NL>(x);
		}
		__device__ static __nv_fp8_e5m2 from(NL x){
			return __nv_fp8_e5m2(static_cast<float>(x));
		}
	};
#endif

*/
// S3_lut_common.cuh -- macro e dichiarazione primaria usate dalle tabelle e dal valutatore S3.
#pragma once
#include <cstdint>

#ifdef __CUDACC__
  #define S3_HD __host__ __device__
#else
  #define S3_HD
#endif

namespace s3lut {
// Specializzata, per ogni F, dall'header generato S3_tables_<bit>_<F>.cuh
template <int F> struct Tab;
} // namespace s3lut

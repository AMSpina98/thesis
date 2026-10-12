/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>

#include "../../S1.cuh"
#include "../../S2.cuh"
#include "../../S3_lut/S3.cuh"

using S2_16_10 = LNNS<uint16_t, 10>;
using S2_16_11 = LNNS<uint16_t, 11>;
using S2_32_23 = LNNS<uint32_t, 23>;
using S3_16_10 = LNNS_S3<uint16_t, 10>;
using S3_32_21 = LNNS_S3<uint32_t, 21>;

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T> struct SharedMemory
{
    __device__ inline operator T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <> struct SharedMemory<double>
{
    __device__ inline operator double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

template<> struct SharedMemory<S2_16_10>
{
    __device__ inline operator S2_16_10 *()
    {
        extern __shared__ int __smem[];
        return (S2_16_10 *)__smem;
    }

    __device__ inline operator const S2_16_10 *() const
    {
        extern __shared__ int __smem[];
        return (S2_16_10 *)__smem;
    }
};

template<> struct SharedMemory<S2_16_11>
{
    __device__ inline operator S2_16_11 *()
    {
        extern __shared__ int __smem[];
        return (S2_16_11 *)__smem;
    }

    __device__ inline operator const S2_16_11 *() const
    {
        extern __shared__ int __smem[];
        return (S2_16_11 *)__smem;
    }
};

template<> struct SharedMemory<S3_16_10>
{
    __device__ inline operator S3_16_10 *()
    {
        extern __shared__ int __smem[];
        return (S3_16_10 *)__smem;
    }

    __device__ inline operator const S3_16_10 *() const
    {
        extern __shared__ int __smem[];
        return (S3_16_10 *)__smem;
    }
};

template<> struct SharedMemory<S3_32_21>
{
    __device__ inline operator S3_32_21 *()
    {
        extern __shared__ int __smem[];
        return (S3_32_21 *)__smem;
    }

    __device__ inline operator const S3_32_21 *() const
    {
        extern __shared__ int __smem[];
        return (S3_32_21 *)__smem;
    }
};

template<> struct SharedMemory<S2_32_23>
{
    __device__ inline operator S2_32_23 *()
    {
        extern __shared__ int __smem[];
        return (S2_32_23 *)__smem;
    }

    __device__ inline operator const S2_32_23 *() const
    {
        extern __shared__ int __smem[];
        return (S2_32_23 *)__smem;
    }
};

template <> struct SharedMemory<Encapsulator<float>>
{
    __device__ inline operator Encapsulator<float> *()
    {
        extern __shared__ int __smem[];
        return (Encapsulator<float> *)__smem;
    }

    __device__ inline operator const Encapsulator<float> *() const
    {
        extern __shared__ int __smem[];
        return (Encapsulator<float> *)__smem;
    }
};
template <> struct SharedMemory<Encapsulator<double>>
{
    __device__ inline operator Encapsulator<double> *()
    {
        extern __shared__ double __smem_d[];
        return (Encapsulator<double> *)__smem_d;
    }

    __device__ inline operator const Encapsulator<double> *() const
    {
        extern __shared__ double __smem_d[];
        return (Encapsulator<double> *)__smem_d;
    }
};

/*
    This version uses sequential addressing -- no divergence or bank conflicts.
*/
template <class T> __global__ void reduce2(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta   = cg::this_thread_block();
    T               *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

extern "C" bool isPow2(unsigned int x);

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T> void reduce(int size, int threads, int blocks, T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);


    // choose which of the optimized versions of reduction to launch
    reduce2<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
}

// Instantiate the reduction function for all types
template void reduce<int>(int size, int threads, int blocks, int *d_idata, int *d_odata);

template void reduce<float>(int size, int threads, int blocks, float *d_idata, float *d_odata);

template void reduce<double>(int size, int threads, int blocks, double *d_idata, double *d_odata);

template void reduce<S2_16_10>(int size, int threads, int blocks, S2_16_10 *d_idata, S2_16_10 *d_odata);

template void reduce<S2_16_11>(int size, int threads, int blocks, S2_16_11 *d_idata, S2_16_11 *d_odata);

template void reduce<S2_32_23>(int size, int threads, int blocks, S2_32_23 *d_idata, S2_32_23 *d_odata);
template void reduce<S3_16_10>(int size, int threads, int blocks, S3_16_10 *d_idata, S3_16_10 *d_odata);
template void reduce<S3_32_21>(int size, int threads, int blocks, S3_32_21 *d_idata, S3_32_21 *d_odata);

template void reduce<Encapsulator<float>>(int size, int threads, int blocks, Encapsulator<float> *d_idata, Encapsulator<float> *d_odata);

template void reduce<Encapsulator<double>>(int size, int threads, int blocks, Encapsulator<double> *d_idata, Encapsulator<double> *d_odata);

#endif // #ifndef _REDUCE_KERNEL_H_

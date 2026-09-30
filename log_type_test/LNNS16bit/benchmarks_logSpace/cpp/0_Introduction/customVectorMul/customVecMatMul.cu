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

/**
 * Matrix-Vector multiplication: y = A * x.
 *
 * This sample is a very basic sample that implements element by element
 * matrix-vector multiplication. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the matrix-vector multiplication of A and x into y. The matrix A has dimensions rows x cols,
 * and the vectors x and y have the same number of elements.
 */

#include "../../LNNS_type.cuh"

using LNNS16_11 = LNNS<uint32_t, 21>; // 16-bit LNNS16_11 with 11 fractional bits

__global__ void matVecMul(const LNNS16_11 *A, const LNNS16_11 *x, LNNS16_11 *y, int rows, int cols)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < rows) {
        LNNS16_11 sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += A[row * cols + col] * x[col];
        }
        y[row] = sum;
    }
}

/**
 * Host main routine
 */
int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int    rows = 5000;
    int    cols = 5000;
    size_t sizeA        = rows * cols * sizeof(LNNS16_11);
    size_t sizeB        = cols * sizeof(LNNS16_11);
    size_t sizeC        = rows * sizeof(LNNS16_11);
    printf("[Matrix-Vector multiplication of %d x %d elements]\n", rows, cols);

    // Allocate the host input vector A
    LNNS16_11 *h_A = (LNNS16_11 *)malloc(sizeA);

    // Allocate the host input vector B
    LNNS16_11 *h_B = (LNNS16_11 *)malloc(sizeB);

    // Allocate the host output vector C
    LNNS16_11 *h_C = (LNNS16_11 *)malloc(sizeC);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < rows * cols; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < cols; ++i) {
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate the device input vector A
    LNNS16_11 *d_A = NULL;
    err        = cudaMalloc((void **)&d_A, sizeA);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    LNNS16_11 *d_B = NULL;
    err        = cudaMalloc((void **)&d_B, sizeB);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    LNNS16_11 *d_C = NULL;
    err        = cudaMalloc((void **)&d_C, sizeC);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input
    // vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Matrix-Vector Multiplication CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid   = (rows + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    matVecMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch matVecMul kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    /*
    // Verify that the result vector is correct
    for (int i = 0; i < rows; ++i) {
        LNNS16_11 res = h_C[i];
        for (int j = 0; j < cols; ++j) {
            res -= h_A[i * cols + j] * h_B[j];
        }
        if (fabs(res.toFloat()) > 1e-5) {
            fprintf(stderr, "Result verification failed at row %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }*/

    for (int i = 0; i < rows; ++i) {
        float expected = 0.0f;

        for (int j = 0; j < cols; ++j) {
            expected += h_A[i * cols + j].toFloat() *
                        h_B[j].toFloat();
        }

        float actual = h_C[i].toFloat();

        float error = fabsf(actual - expected);

        if (error > (1e-3f*fabsf(actual))) {
            fprintf(stderr,
                    "Row %d: expected=%f actual=%f error=%f\n",
                    i, expected, actual, error);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}

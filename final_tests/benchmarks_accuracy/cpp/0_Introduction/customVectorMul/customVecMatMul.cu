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
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "../../utility.cu"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

#include "../../S1.cuh"
#include "../../S2.cuh"
#include "../../S3_lut/S3.cuh"

using S3_16_10 = LNNS_S3<uint16_t, 10>;
using S3_32_21 = LNNS_S3<uint32_t, 21>;

using S2_16_10 = LNNS<uint16_t, 10>;
using S2_16_11 = LNNS<uint16_t, 11>;
using S2_32_23 = LNNS<uint32_t, 23>;
using EncFp = Encapsulator<float>;
using EncDb = Encapsulator<double>;

template <typename T>
__global__ void matVecMul(const T *A, const T *x, T *y, int rows, int cols)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < rows) {
        T sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += A[row * cols + col] * x[col];
        }
        y[row] = sum;
    }
}

__global__ void matVecMulFloat(const float *A, const float *x, float *y, int rows, int cols)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += A[row * cols + col] * x[col];
        }
        y[row] = sum;
    }
}


/**
 * Host main routine
 */

template <typename T>
void runTest(const std::string& type){
// Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int    rows = 4000;
    int    cols = 4000;


    size_t sizeA        = rows * cols * sizeof(T);
    size_t sizeA_float  = rows * cols * sizeof(float);
    size_t sizeB        = cols * sizeof(T);
    size_t sizeB_float  = cols * sizeof(float);
    size_t sizeC        = rows * sizeof(T);
    size_t sizeC_float  = rows * sizeof(float);
    printf("[Matrix-Vector multiplication of %d x %d elements]\n", rows, cols);

    // Allocate the host input vector A
    T *h_A = (T *)malloc(sizeA);

    float *h_A_float = (float *)malloc(sizeA_float);

    // Allocate the host input vector B
    T *h_B = (T *)malloc(sizeB);

    float *h_B_float = (float *)malloc(sizeB_float);

    // Allocate the host output vector C
    T *h_C = (T *)malloc(sizeC);

    float *h_C_float = (float *)malloc(sizeC_float);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < rows * cols; ++i) {
        float a = rand() / (float)RAND_MAX;
        h_A[i] = a;
        h_A_float[i] = a;
    }
    for (int i = 0; i < cols; ++i) {
        float b = rand() / (float)RAND_MAX;
        h_B[i] = b;
        h_B_float[i] = b;
    }

    // Allocate the device input vector A
    T *d_A = NULL;
    err        = cudaMalloc((void **)&d_A, sizeA);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_A_float = NULL;
    err        = cudaMalloc((void **)&d_A_float, sizeA_float);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A_float (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    T *d_B = NULL;
    err        = cudaMalloc((void **)&d_B, sizeB);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_B_float = NULL;
    err        = cudaMalloc((void **)&d_B_float, sizeB_float);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B_float (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    T *d_C = NULL;
    err        = cudaMalloc((void **)&d_C, sizeC);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *d_C_float = NULL;
    err        = cudaMalloc((void **)&d_C_float, sizeC_float);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C_float (error code %s)!\n", cudaGetErrorString(err));
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

    err = cudaMemcpy(d_A_float, h_A_float, sizeA_float, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A_float from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B_float, h_B_float, sizeB_float, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B_float from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Matrix-Vector Multiplication CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid   = (rows + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    
    cudaEvent_t startE, stopE;
	cudaEventCreate(&startE);
	cudaEventCreate(&stopE);
    float elapsed =  0.0f;
    float elapsed_float = 0.0f;

    int nIter = 300;

    for(int i = 0; i < 300; ++i) {    
        cudaEventRecord(startE);

        matVecMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

        cudaEventRecord(stopE);
        cudaEventSynchronize(stopE);
        float elapsed_temp;
        cudaEventElapsedTime(&elapsed_temp, startE, stopE);
        elapsed += elapsed_temp;


        err = cudaGetLastError();

        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch matVecMul kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        printf("CUDA kernel float launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

        cudaEventRecord(startE);

        matVecMulFloat<<<blocksPerGrid, threadsPerBlock>>>(d_A_float, d_B_float, d_C_float, rows, cols);

        
        cudaEventRecord(stopE);
        cudaEventSynchronize(stopE);
        float elapsed_temp_float;
        cudaEventElapsedTime(&elapsed_temp_float, startE, stopE);
        elapsed_float += elapsed_temp_float;


        err = cudaGetLastError();

        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch matVecMul kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    elapsed /= nIter;
    elapsed_float /= nIter;

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_C_float, d_C_float, sizeC_float, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C_float from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Test PASSED\n");


    float maxAbsErrorA = 0.0f;
    float maxRelErrorA = 0.0f;
    float avgAbsErrorA = 0.0f;
    float avgRelErrorA = 0.0f;
    float maxAbsErrorB = 0.0f;
    float maxRelErrorB = 0.0f;
    float avgAbsErrorB = 0.0f;
    float avgRelErrorB = 0.0f;
    float maxAbsErrorC = 0.0f;
    float maxRelErrorC = 0.0f;
    float avgAbsErrorC = 0.0f;
    float avgRelErrorC = 0.0f;
    int elementsA = rows * cols;
    int elementsB = cols;
    int elementsC = rows;
    int countA = 0;
    int countB = 0;
    int countC = 0;
    float RSME_A = 0.0f;
    float RSME_B = 0.0f;
    float RSME_C = 0.0f;

    float threshold = 1e-3f;

    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            float errorA = fabsf((float)h_A[i * cols + j] - h_A_float[i * cols + j]);
            float den = fabsf(h_A_float[i * cols + j]) != 0.0f ? fabsf(h_A_float[i * cols + j]) : 1.0f;
            float relErrorA = fabsf(errorA / den);
            if(relErrorA > maxRelErrorA) maxRelErrorA = relErrorA;
            avgRelErrorA += relErrorA;
            if(errorA > maxAbsErrorA) maxAbsErrorA = errorA;
            avgAbsErrorA += errorA;
            RSME_A += errorA * errorA;
            if(errorA > fabsf(h_A_float[i * cols + j]*threshold)) countA++;
        }
    }
    
    avgAbsErrorA /= elementsA;
    RSME_A = sqrt(RSME_A / elementsA);
    avgRelErrorA /= elementsA;
    
    for(int i = 0; i < cols; ++i) {
        float errorB = fabsf((float)h_B[i] - h_B_float[i]);
        float den = fabsf(h_B_float[i]) != 0.0f ? fabsf(h_B_float[i]) : 1.0f;
        float relErrorB = fabsf(errorB / den);
        if(relErrorB > maxRelErrorB) maxRelErrorB = relErrorB;
        avgRelErrorB += relErrorB;
        if(errorB > maxAbsErrorB) maxAbsErrorB = errorB;
        avgAbsErrorB += errorB;
        RSME_B += errorB * errorB;
        if(errorB > fabsf(h_B_float[i]*threshold)) countB++;
    }
    
    avgAbsErrorB /= elementsB;
    RSME_B = sqrt(RSME_B / elementsB);
    avgRelErrorB /= elementsB;
    
    for(int i = 0; i < rows; ++i) {
        float errorC = fabsf((float)h_C[i] - h_C_float[i]);
        float den = fabsf(h_C_float[i]) != 0.0f ? fabsf(h_C_float[i]) : 1.0f;
        float relErrorC = fabsf(errorC / den);
        if(relErrorC > maxRelErrorC) maxRelErrorC = relErrorC;
        avgRelErrorC += relErrorC;
        if(errorC > maxAbsErrorC) maxAbsErrorC = errorC;
        avgAbsErrorC += errorC;
        RSME_C += errorC * errorC;
        if(errorC > fabsf(h_C_float[i]*threshold)) countC++;
    }
    avgAbsErrorC /= elementsC;
    RSME_C = sqrt(RSME_C / elementsC);
    avgRelErrorC /= elementsC;

    saveResultsCSV(
        "vetmatmul_A_results.csv",
        type,
        elementsA,
        threshold,
        maxAbsErrorA,
        avgAbsErrorA,
        RSME_A,
        maxRelErrorA,
        avgRelErrorA,
        countA
    );

    saveResultsCSV(
        "vetmatmul_B_results.csv",
        type,
        elementsB,
        threshold,
        maxAbsErrorB,
        avgAbsErrorB,
        RSME_B,
        maxRelErrorB,
        avgRelErrorB,
        countB
    );

    saveResultsCSV(
        "vetmatmul_C_results.csv",
        type,
        elementsC,
        threshold,
        maxAbsErrorC,
        avgAbsErrorC,
        RSME_C,
        maxRelErrorC,
        avgRelErrorC,
        countC
    );
    
    
    saveTimingCSV(
        "vetmatmul_timing.csv",
        type,
        elementsC,
        elapsed,
        elapsed_float
    );



    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_A_float);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector A_float (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B_float);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B_float (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C_float);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector C_float (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    free(h_A_float);
    free(h_B_float);
    free(h_C_float);

    printf("Done\n");
}


int main(int argc, char **argv)
{
    int testNum = 0;
    if (checkCmdLineFlag(argc, (const char **)argv, "testNum")) {
        testNum = getCmdLineArgumentInt(argc, (const char **)argv, "testNum");
    }

    switch(testNum){
        case 0: runTest<float>("float"); break;
        case 1: runTest<double>("double"); break;
        case 2: runTest<EncFp>("EncFp"); break;
        case 3: runTest<EncDb>("EncDb"); break;
        case 4: runTest<S2_16_10>("S2_16_10"); break;
        case 5: runTest<S2_16_11>("S2_16_11"); break;
        case 6: runTest<S2_32_23>("S2_32_23"); break;
        case 7: runTest<S3_16_10>("S3_16_10"); break;
        case 8: runTest<S3_32_21>("S3_32_21"); break;
        default:
            printf("Invalid test number: %d\n", testNum);
            exit(EXIT_FAILURE);
    }


    return 0;
    
}

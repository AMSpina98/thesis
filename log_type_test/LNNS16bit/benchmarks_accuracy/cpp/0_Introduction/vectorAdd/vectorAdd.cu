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
#include <vector>
#include <algorithm>
#include <cmath>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "../utility.cu"
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

 #include "../LNNS_type.cuh"

 using LNNS32_23 = LNNS<uint32_t, 23>; // 32-bit LNNS32_23 with 23 fractional bits

__global__ void vectorAdd(const LNNS32_23 *A, const LNNS32_23 *B, LNNS32_23 *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i] + 0.0f;
    }
}

__global__ void originalVectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i] + 0.0f;
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
    int    numElements = 50000;
    size_t size        = numElements * sizeof(LNNS32_23);
    size_t size_float  = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    LNNS32_23 *h_A = (LNNS32_23 *)malloc(size);
    float *h_A_float = (float *)malloc(size_float);

    // Allocate the host input vector B
    LNNS32_23 *h_B = (LNNS32_23 *)malloc(size);
    float *h_B_float = (float *)malloc(size_float);

    // Allocate the host output vector C
    LNNS32_23 *h_C = (LNNS32_23 *)malloc(size);
    float *h_C_float = (float *)malloc(size_float);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    if(h_A_float == NULL || h_B_float == NULL || h_C_float == NULL) {
        fprintf(stderr, "Failed to allocate host float vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        float a = rand() / (float)RAND_MAX;
        float b = rand() / (float)RAND_MAX;
        h_A[i] = a;
        h_A_float[i] = a;
        h_B[i] = b;
        h_B_float[i] = b;
    }

    // Allocate the device input vector A
    LNNS32_23 *d_A = NULL;
    err        = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *d_A_float = NULL;
    err        = cudaMalloc((void **)&d_A_float, size_float);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device float vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    LNNS32_23 *d_B = NULL;
    err        = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_B_float = NULL;
    err        = cudaMalloc((void **)&d_B_float, size_float);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device float vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    LNNS32_23 *d_C = NULL;
    err        = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *d_C_float = NULL;
    err        = cudaMalloc((void **)&d_C_float, size_float);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device float vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input
    // vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_A_float, h_A_float, size_float, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy float vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B_float, h_B_float, size_float, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy float vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }










    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid   = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    
    cudaEvent_t startE, stopE;
	cudaEventCreate(&startE);
	cudaEventCreate(&stopE);

    int nWarmup = 200;   // scartati dalla misura: portano il clock GPU a regime
    int nIter   = 300000;

    // --- Warm-up (non cronometrato) ---
    for (int i = 0; i < nWarmup; i++) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        originalVectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A_float, d_B_float, d_C_float, numElements);
    }
    cudaDeviceSynchronize();

    // --- Run cronometrati: si tiene traccia di ogni singolo campione, non solo
    // della somma, per poter calcolare mediana/deviazione standard/p95 oltre
    // alla media (la sola media può restare ingannevole anche su tanti run se
    // la distribuzione ha una coda lunga di outlier). ---
    std::vector<float> samplesLNNS(nIter);
    std::vector<float> samplesFloat(nIter);

    for (int i = 0; i < nIter; i++) {
        cudaEventRecord(startE);

        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

        cudaEventRecord(stopE);
        cudaEventSynchronize(stopE);
        cudaEventElapsedTime(&samplesLNNS[i], startE, stopE);

        err = cudaGetLastError();

        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        cudaEventRecord(startE);

        originalVectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A_float, d_B_float, d_C_float, numElements);

        cudaEventRecord(stopE);
        cudaEventSynchronize(stopE);
        cudaEventElapsedTime(&samplesFloat[i], startE, stopE);

        err = cudaGetLastError();

        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch originalVectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    auto computeStats = [](std::vector<float> samples, float& mean, float& median, float& stddev, float& p95) {
        size_t n = samples.size();
        double sum = 0.0;
        for (float v : samples) sum += v;
        mean = static_cast<float>(sum / n);

        double sqSum = 0.0;
        for (float v : samples) sqSum += (double(v) - mean) * (double(v) - mean);
        stddev = static_cast<float>(std::sqrt(sqSum / n));

        std::sort(samples.begin(), samples.end());
        median = samples[n / 2];
        p95    = samples[static_cast<size_t>(0.95 * n)];
    };

    float meanLNNS, medianLNNS, stddevLNNS, p95LNNS;
    float meanFloat, medianFloat, stddevFloat, p95Float;
    computeStats(samplesLNNS, meanLNNS, medianLNNS, stddevLNNS, p95LNNS);
    computeStats(samplesFloat, meanFloat, medianFloat, stddevFloat, p95Float);

    float elapsed = meanLNNS;
    float elapsed_float = meanFloat;
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the float result vector from device to host
    err = cudaMemcpy(h_C_float, d_C_float, size_float, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy float vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



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
    int elementsA = numElements;
    int elementsB = numElements;
    int elementsC = numElements;
    int countA = 0;
    int countB = 0;
    int countC = 0;
    float RSME_A = 0.0f;
    float RSME_B = 0.0f;
    float RSME_C = 0.0f;

    float threshold = 1e-3f;

    for(int i = 0; i < elementsA; ++i) {
        float errorA = fabsf(h_A[i].toFloat() - h_A_float[i]);
        float den = fabsf(h_A_float[i]) != 0.0f ? fabsf(h_A_float[i]) : 1.0f;
        float relErrorA = fabsf(errorA / den);
        if(relErrorA > maxRelErrorA) maxRelErrorA = relErrorA;
        avgRelErrorA += relErrorA;
        if(errorA > maxAbsErrorA) maxAbsErrorA = errorA;
        avgAbsErrorA += errorA;
        RSME_A += errorA * errorA;
        if(errorA > fabsf(h_A_float[i]*threshold)) countA++;
    }
    
    avgAbsErrorA /= elementsA;
    RSME_A = sqrt(RSME_A / elementsA);
    avgRelErrorA /= elementsA;
    
    for(int i = 0; i < elementsB; ++i) {
        float errorB = fabsf(h_B[i].toFloat() - h_B_float[i]);
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
    
    for(int i = 0; i < elementsC; ++i) {
        float errorC = fabsf(h_C[i].toFloat() - h_C_float[i]);
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
        "vectorAdd_A_results.csv",
        "LNNS16_21",
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
        "vectorAdd_B_results.csv",
        "LNNS16_21",
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
        "vectorAdd_C_results.csv",
        "LNNS16_21",
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
        "vectorAdd_timing.csv",
        "LNNS16_21",
        numElements,
        elapsed,
        elapsed_float
    );

    saveTimingStatsCSV(
        "vectorAdd_timing_stats.csv",
        "LNNS16_21",
        numElements,
        meanLNNS, medianLNNS, stddevLNNS, p95LNNS,
        meanFloat, medianFloat, stddevFloat, p95Float
    );





    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_A_float);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device float vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B_float);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device float vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C_float);
     if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device float vector C (error code %s)!\n", cudaGetErrorString(err));
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
    return 0;
}

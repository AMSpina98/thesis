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
    Parallel reduction

    This sample shows how to perform a reduction operation on an array of values
    to produce a single value.

    MODIFICA: oltre al pass/fail contro un riferimento calcolato in T (Kahan
    summation nel dominio di T -- confronto "auto-referenziale", vedi nota
    sotto), ora viene calcolato anche un riferimento CPU in DOUBLE
    (reduceCPUDouble), indipendente dalla precisione di T, e le metriche di
    errore (assoluto/relativo, RMSE degenere a un solo valore, conteggio
    sopra soglia) vengono salvate su CSV con lo stesso formato degli altri
    benchmark (saveResultsCSV / saveTimingCSV, da "../utility.cu").

    Nota sul riferimento in T (reduceCPU<T>, lasciato invariato): la somma di
    Kahan e' una correzione tarata sul modello di arrotondamento IEEE-754
    float/double; su un T quantizzato (LNNS) o rerouted (Encapsulator) il
    termine di compensazione non ha lo stesso significato, e il confronto
    resta "T vs T" (stessa precisione), non "T vs verita'". Per questo il
    riferimento usato per le nuove metriche di accuratezza e' calcolato in
    double, non riusando reduceCPU<T>.

    COMMAND LINE ARGUMENTS

    "--shmoo":         Test performance for 1 to 32M elements with each of the 7
   different kernels
    "--n=<N>":         Specify the number of elements to reduce (default
   1048576)
    "--threads=<N>":   Specify the number of threads per block (default 128)
    "--kernel=<N>":    Specify which kernel to run (0-6, default 6)
    "--maxblocks=<N>": Specify the maximum number of thread blocks to launch
   (kernel 6 only, default 64)
    "--cpufinal":      Read back the per-block results and do final sum of block
   sums on CPU (default false)
    "--cputhresh=<N>": The threshold of number of blocks sums below which to
   perform a CPU final reduction (default 1)
    "-type=<T>":       The datatype for the reduction, where T is "int",
   "float", or "double" (default int)
*/

// CUDA Runtime
#include <cuda_runtime.h>
#include "../../S2.cuh"
#include "../../S1.cuh"

using S2_16_10 = LNNS<uint16_t, 10>;
using S2_16_11 = LNNS<uint16_t, 11>;
using S2_32_21 = LNNS<uint32_t, 21>;


// Utilities and system includes
#include <algorithm>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "../../utility.cu"

// includes, project
#include "reduction.h"
#include "reduction_kernel.cu"

enum ReduceType { REDUCE_INT, REDUCE_FLOAT, REDUCE_DOUBLE, REDUCE_S2_16_10, REDUCE_S2_16_11, REDUCE_S2_32_21, REDUCE_ENCAPSULATOR_FLOAT, REDUCE_ENCAPSULATOR_DOUBLE };

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
template <class T> bool runTest(int argc, char **argv, ReduceType datatype);

#define MAX_BLOCK_DIM_SIZE 65535
constexpr int BENCHMARK_ITERATIONS = 10000;

#ifdef WIN32
#define strcasecmp strcmpi
#endif

extern "C" bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

const char *getReduceTypeString(const ReduceType type)
{
    switch (type) {
    case REDUCE_INT:
        return "int";
    case REDUCE_FLOAT:
        return "float";
    case REDUCE_DOUBLE:
        return "double";
    case REDUCE_S2_16_10:
        return "S2_16_10";
    case REDUCE_S2_16_11:
        return "S2_16_11";
    case REDUCE_S2_32_21:
        return "S2_32_21";
    case REDUCE_ENCAPSULATOR_FLOAT:
        return "Encapsulator<float>";
    case REDUCE_ENCAPSULATOR_DOUBLE:
        return "Encapsulator<double>";
    default:
        return "unknown";
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);

    char *typeInput = 0;
    getCmdLineArgumentString(argc, (const char **)argv, "type", &typeInput);

    ReduceType datatype = REDUCE_INT;

    if (0 != typeInput) {
        if (!strcasecmp(typeInput, "float")) {
            datatype = REDUCE_FLOAT;
        }
        else if (!strcasecmp(typeInput, "double")) {
            datatype = REDUCE_DOUBLE;
        }
        else if (!strcasecmp(typeInput, "S2_16_10")) {
            datatype = REDUCE_S2_16_10;
        }
        else if (!strcasecmp(typeInput, "S2_16_11")) {
            datatype = REDUCE_S2_16_11;
        }
        else if (!strcasecmp(typeInput, "S2_32_21")) {
            datatype = REDUCE_S2_32_21;
        }
        else if (!strcasecmp(typeInput, "Encapsulator<float>")) {
            datatype = REDUCE_ENCAPSULATOR_FLOAT;
        }
        else if (!strcasecmp(typeInput, "Encapsulator<double>")) {
            datatype = REDUCE_ENCAPSULATOR_DOUBLE;
        }
        else if (strcasecmp(typeInput, "int")) {
            printf("Type %s is not recognized. Using default type int.\n\n", typeInput);
        }
    }

    cudaDeviceProp deviceProp;
    int            dev;

    dev = findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    printf("Using Device %d: %s\n\n", dev, deviceProp.name);
    checkCudaErrors(cudaSetDevice(dev));

    printf("Reducing array of type %s\n\n", getReduceTypeString(datatype));

    bool bResult = false;

    switch (datatype) {
    default:
    case REDUCE_INT:
        bResult = runTest<int>(argc, argv, datatype);
        break;

    case REDUCE_FLOAT:
        bResult = runTest<float>(argc, argv, datatype);
        break;

    case REDUCE_DOUBLE:
        bResult = runTest<double>(argc, argv, datatype);
        break;
    case REDUCE_S2_16_10:
        bResult = runTest<S2_16_10>(argc, argv, datatype);
        break;
    case REDUCE_S2_16_11:
        bResult = runTest<S2_16_11>(argc, argv, datatype);
        break;
    case REDUCE_S2_32_21:
        bResult = runTest<S2_32_21>(argc, argv, datatype);
        break;
    case REDUCE_ENCAPSULATOR_FLOAT:
        bResult = runTest<Encapsulator<float>>(argc, argv, datatype);
        break;
    case REDUCE_ENCAPSULATOR_DOUBLE:
        bResult = runTest<Encapsulator<double>>(argc, argv, datatype);
        break;
    }

    printf(bResult ? "Test passed\n" : "Test failed!\n");
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
template <class T> T reduceCPU(T *data, int size)
{
    T sum = data[0];
    T c   = (T)0.0;

    for (int i = 1; i < size; i++) {
        T y = data[i] - c;
        T t = sum + y;
        c   = (t - sum) - y;
        sum = t;
    }

    return sum;
}

////////////////////////////////////////////////////////////////////////////////
//! Riferimento CPU in DOUBLE, indipendente dalla precisione di T -- usato
//! per le metriche di accuratezza (vedi nota in testa al file). Richiede
//! che T sia convertibile a double (operator double(), gia' presente su
//! int/float/double nativi e aggiunto a Encapsulator<T>/LNNS<T,P>).
////////////////////////////////////////////////////////////////////////////////
template <class T> double reduceCPUDouble(T *data, int size)
{
    double sum = (double)data[0];
    double c   = 0.0;

    for (int i = 1; i < size; i++) {
        double y = (double)data[i] - c;
        double t = sum + y;
        c        = (t - sum) - y;
        sum      = t;
    }

    return sum;
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

#ifndef MIN
#define MIN(x, y) ((x < y) ? x : y)
#endif

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction
// kernel. reduce2 (l'unico kernel rimasto, un elemento per thread) usa
// sempre la formula "threads = min(nextPow2(n), maxThreads)".
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    // get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int            device;
    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));

    threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
    blocks  = (n + threads - 1) / threads;

    if ((float)threads * blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0]) {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as "
               "%d (original %d)\n",
               blocks,
               prop.maxGridSize[0],
               threads * 2,
               threads);

        blocks /= 2;
        threads *= 2;
    }
}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
template <class T>
T benchmarkReduce(int                 n,
                  int                 numThreads,
                  int                 numBlocks,
                  int                 maxThreads,
                  int                 maxBlocks,
                  int                 testIterations,
                  bool                cpuFinalReduction,
                  int                 cpuFinalThreshold,
                  float              &avgTimeMs,
                  T                  *h_odata,
                  T                  *d_idata,
                  T                  *d_odata)
{
    T    gpu_result   = 0;
    bool needReadBack = true;

    T *d_intermediateSums;
    checkCudaErrors(cudaMalloc((void **)&d_intermediateSums, sizeof(T) * numBlocks));

    cudaEvent_t startE, stopE;
    cudaEventCreate(&startE);
    cudaEventCreate(&stopE);

    double totalTimeMs = 0.0;

    for (int i = 0; i < testIterations; ++i) {
        gpu_result = 0;

        cudaDeviceSynchronize();
        cudaEventRecord(startE);

        // execute the kernel
        reduce<T>(n, numThreads, numBlocks, d_idata, d_odata);
        getLastCudaError("Kernel execution failed");

        if (cpuFinalReduction) {
            checkCudaErrors(cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(T), cudaMemcpyDeviceToHost));
            for (int j = 0; j < numBlocks; j++) {
                gpu_result += h_odata[j];
            }
            needReadBack = false;
        }
        else {
            int s = numBlocks;
            while (s > cpuFinalThreshold) {
                int threads = 0, blocks = 0;
                getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);
                checkCudaErrors(cudaMemcpy(d_intermediateSums, d_odata, s * sizeof(T), cudaMemcpyDeviceToDevice));
                reduce<T>(s, threads, blocks, d_intermediateSums, d_odata);
                s = (s + threads - 1) / threads;
            }
            if (s > 1) {
                checkCudaErrors(cudaMemcpy(h_odata, d_odata, s * sizeof(T), cudaMemcpyDeviceToHost));
                for (int j = 0; j < s; j++) {
                    gpu_result += h_odata[j];
                }
                needReadBack = false;
            }
        }

        cudaEventRecord(stopE);
        cudaEventSynchronize(stopE);

        float iterMs = 0.0f;
        cudaEventElapsedTime(&iterMs, startE, stopE);
        totalTimeMs += iterMs;
    }

    avgTimeMs = static_cast<float>(totalTimeMs / testIterations);

    if (needReadBack) {
        checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));
    }

    cudaEventDestroy(startE);
    cudaEventDestroy(stopE);
    checkCudaErrors(cudaFree(d_intermediateSums));
    return gpu_result;
}


////////////////////////////////////////////////////////////////////////////////
// Misura il tempo medio di riduzione per un tipo U, usata come baseline
// "float" per calcolare l'overhead degli altri tipi nel CSV di timing.
////////////////////////////////////////////////////////////////////////////////
template <class U>
double measureReduceTime(int size, int numThreads, int numBlocks, int maxThreads, int maxBlocks, int testIterations)
{
    unsigned int bytes = size * sizeof(U);

    U *h_idata = (U *)malloc(bytes);
    for (int i = 0; i < size; i++) {
        h_idata[i] = (U)((rand() & 0xFF) / (float)RAND_MAX);
    }

    U *h_odata = (U *)malloc(numBlocks * sizeof(U));

    U *d_idata = NULL;
    U *d_odata = NULL;
    checkCudaErrors(cudaMalloc((void **)&d_idata, bytes));
    checkCudaErrors(cudaMalloc((void **)&d_odata, numBlocks * sizeof(U)));

    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_odata, h_idata, numBlocks * sizeof(U), cudaMemcpyHostToDevice));

    // warm-up
    reduce<U>(size, numThreads, numBlocks, d_idata, d_odata);

    float avgTimeMs = 0.0f;
    benchmarkReduce<U>(size, numThreads, numBlocks, maxThreads, maxBlocks,
                        testIterations, false, 1,
                        avgTimeMs, h_odata, d_idata, d_odata);

    free(h_idata);
    free(h_odata);
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    return avgTimeMs * 1e-3;  // secondi, come prima
}


////////////////////////////////////////////////////////////////////////////////
// This function calls benchmarkReduce multiple times for a range of array sizes
// and prints a report in CSV (comma-separated value) format that can be used
// for generating a "shmoo" plot showing the performance for each kernel
// variation over a wide range of input sizes.
////////////////////////////////////////////////////////////////////////////////
template <class T> void shmoo(int minN, int maxN, int maxThreads, int maxBlocks, ReduceType datatype)
{
    // create random input data on CPU
    unsigned int bytes = maxN * sizeof(T);

    T *h_idata = (T *)malloc(bytes);

    for (int i = 0; i < maxN; i++) {
        // Keep the numbers small so we don't get truncation error in the sum
        if (datatype == REDUCE_INT) {
            h_idata[i] = (T)(rand() & 0xFF);
        }
        else {
            h_idata[i] = (rand() & 0xFF) / (float)RAND_MAX;
        }
    }

    int maxNumBlocks = MIN(maxN / maxThreads, MAX_BLOCK_DIM_SIZE);

    // allocate mem for the result on host side
    T *h_odata = (T *)malloc(maxNumBlocks * sizeof(T));

    // allocate device memory and data
    T *d_idata = NULL;
    T *d_odata = NULL;

    checkCudaErrors(cudaMalloc((void **)&d_idata, bytes));
    checkCudaErrors(cudaMalloc((void **)&d_odata, maxNumBlocks * sizeof(T)));

    // copy data directly to device memory
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_odata, h_idata, maxNumBlocks * sizeof(T), cudaMemcpyHostToDevice));

    reduce<T>(maxN, maxThreads, maxNumBlocks, d_idata, d_odata);


    int testIterations = BENCHMARK_ITERATIONS;

    float avgTimeMs = 0.0f;


    for (int i = minN; i <= maxN; i *= 2) {
        int numBlocks  = 0;
        int numThreads = 0;
        getNumBlocksAndThreads(i, maxBlocks, maxThreads, numBlocks, numThreads);

        float reduceTime;

        if (numBlocks <= MAX_BLOCK_DIM_SIZE) {
            benchmarkReduce(i,
                            numThreads,
                            numBlocks,
                            maxThreads,
                            maxBlocks,
                            testIterations,
                            false,
                            1,
                            avgTimeMs,
                            h_odata,
                            d_idata,
                            d_odata);
            reduceTime = avgTimeMs * 1e-3;
        }
        else {
            reduceTime = -1.0;
        }

        printf(", %.5f", reduceTime);
    }

    // cleanup
    free(h_idata);
    free(h_odata);

    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));
}

////////////////////////////////////////////////////////////////////////////////
// The main function which runs the reduction test.
////////////////////////////////////////////////////////////////////////////////
template <class T> bool runTest(int argc, char **argv, ReduceType datatype)
{
    int  size              = 1 << 24; // number of elements to reduce
    int  maxThreads        = 256;     // number of threads per block
    int  maxBlocks         = 64;
    bool cpuFinalReduction = false;
    int  cpuFinalThreshold = 1;

    if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
        size = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "threads")) {
        maxThreads = getCmdLineArgumentInt(argc, (const char **)argv, "threads");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "maxblocks")) {
        maxBlocks = getCmdLineArgumentInt(argc, (const char **)argv, "maxblocks");
    }

    printf("%d elements\n", size);
    printf("%d threads (max)\n", maxThreads);

    cpuFinalReduction = checkCmdLineFlag(argc, (const char **)argv, "cpufinal");

    if (checkCmdLineFlag(argc, (const char **)argv, "cputhresh")) {
        cpuFinalThreshold = getCmdLineArgumentInt(argc, (const char **)argv, "cputhresh");
    }

    bool runShmoo = checkCmdLineFlag(argc, (const char **)argv, "shmoo");

    if (runShmoo) {
        shmoo<T>(1, 33554432, maxThreads, maxBlocks, datatype);
    }
    else {
        // create random input data on CPU
        unsigned int bytes = size * sizeof(T);

        T *h_idata = (T *)malloc(bytes);

        for (int i = 0; i < size; i++) {
            // Keep the numbers small so we don't get truncation error in the sum
            if (datatype == REDUCE_INT) {
                h_idata[i] = (rand() & 0xFF);
            }
            else {
                h_idata[i] = (rand() & 0xFF) / (float)RAND_MAX;
            }
        }

        int numBlocks  = 0;
        int numThreads = 0;
        getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

        if (numBlocks == 1) {
            cpuFinalThreshold = 1;
        }

        // allocate mem for the result on host side
        T *h_odata = (T *)malloc(numBlocks * sizeof(T));

        printf("%d blocks\n\n", numBlocks);

        // allocate device memory and data
        T *d_idata = NULL;
        T *d_odata = NULL;

        checkCudaErrors(cudaMalloc((void **)&d_idata, bytes));
        checkCudaErrors(cudaMalloc((void **)&d_odata, numBlocks * sizeof(T)));

        // copy data directly to device memory
        checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_odata, h_idata, numBlocks * sizeof(T), cudaMemcpyHostToDevice));

        // warm-up
        reduce<T>(size, numThreads, numBlocks, d_idata, d_odata);

        int testIterations = BENCHMARK_ITERATIONS;

        float avgTimeMs = 0.0f;

        T gpu_result = benchmarkReduce<T>(size, numThreads, numBlocks, maxThreads, maxBlocks,
                                        testIterations, cpuFinalReduction, cpuFinalThreshold,
                                        avgTimeMs, h_odata, d_idata, d_odata);

        double reduceTime = avgTimeMs * 1e-3;

        printf("Reduction, Throughput = %.4f GB/s, Time = %.5f s, Size = %u Elements, "
               "NumDevsUsed = %d, Workgroup = %u\n",
               1.0e-9 * ((double)bytes) / reduceTime,
               reduceTime,
               size,
               1,
               numThreads);

        // compute reference solution (T-domain, Kahan) -- usato per il
        // pass/fail storico, invariato
        T cpu_result = reduceCPU<T>(h_idata, size);

        int    precision = 0;
        double threshold = 0;
        double diff      = 0;

        if (datatype == REDUCE_INT) {
            printf("\nGPU result = %d\n", (int)gpu_result);
            printf("CPU result = %d\n\n", (int)cpu_result);
        }
        else {
            if (datatype == REDUCE_FLOAT) {
                precision = 8;
                threshold = 1e-8 * size;
            }
            else if (datatype == REDUCE_S2_16_10) {
                precision = 8;
                threshold = 1e-4 * size;
            }
            else if (datatype == REDUCE_S2_16_11) {
                precision = 8;
                threshold = 1e-4 * size;
            }
            else if (datatype == REDUCE_S2_32_21) {
                precision = 12;
                threshold = 1e-8 * size;
            }
            else if (datatype == REDUCE_ENCAPSULATOR_FLOAT) {
                precision = 8;
                threshold = 1e-8 * size;
            }
            else if (datatype == REDUCE_ENCAPSULATOR_DOUBLE) {
                precision = 8;
                threshold = 1e-8 * size;
            }
            else {
                precision = 12;
                threshold = 1e-12 * size;
            }

            printf("\nGPU result = %.*f\n", precision, (double)gpu_result);
            printf("CPU result = %.*f\n\n", precision, (double)cpu_result);

            diff = fabs((double)gpu_result - (double)cpu_result);
        }

        // -----------------------------------------------------------------
        // Metriche di accuratezza vs riferimento CPU in DOUBLE (indipendente
        // da T, vedi nota in testa al file). La riduzione produce un solo
        // valore scalare, quindi qui "max" ed "avg" (e la RMSE) coincidono
        // tutti con l'unico errore disponibile -- non e' un errore di
        // copia/incolla dal pattern elemento-per-elemento degli altri
        // benchmark, e' la forma corretta per una singola osservazione.
        // -----------------------------------------------------------------
        if (datatype != REDUCE_INT) {
            double cpuResultDouble = reduceCPUDouble<T>(h_idata, size);
            double gpuResultDouble = (double)gpu_result;
            double absError        = fabs(gpuResultDouble - cpuResultDouble);
            double den             = (fabs(cpuResultDouble) > 1e-30) ? fabs(cpuResultDouble) : 1.0;
            double relError        = absError / den;
            int    countOverThresh = (absError > threshold) ? 1 : 0;

            saveResultsCSV(
                "reduction_results.csv",
                std::string(getReduceTypeString(datatype)),
                size,
                threshold,
                (float)absError,
                (float)absError,
                (float)absError,
                (float)relError,
                (float)relError,
                countOverThresh
            );

            // Tempo di riferimento "float" per l'overhead: se T e' gia' float
            // riusa il tempo appena misurato, altrimenti lancia una seconda
            // riduzione identica (stessa size/config) in float, solo per il timing.
            double floatReduceTime = reduceTime;
            if constexpr (!std::is_same<T, float>::value) {
                floatReduceTime = measureReduceTime<float>(size, numThreads, numBlocks, maxThreads, maxBlocks, testIterations);
            }

            saveTimingCSV(
                "reduction_timing.csv",
                std::string(getReduceTypeString(datatype)),
                size,
                (float)reduceTime,
                (float)floatReduceTime
            );
        }
        // cleanup
        free(h_idata);
        free(h_odata);

        checkCudaErrors(cudaFree(d_idata));
        checkCudaErrors(cudaFree(d_odata));

        if (datatype == REDUCE_INT) {
            return (gpu_result == cpu_result);
        }
        else {
            return (diff < threshold);
        }
    }

    return true;
}

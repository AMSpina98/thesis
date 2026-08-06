#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


// =======================================================
// 1. ERROR HANDLING (replacement for helper_cuda / cutil)
// =======================================================

inline void getLastCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA ERROR after %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


// Optional: direct check wrapper (like checkCudaErrors)
#define checkCudaErrors(val)                         \
    do {                                              \
        cudaError_t _err = (val);                    \
        if (_err != cudaSuccess) {                   \
            printf("CUDA Error: %s (line %d)\n",     \
                cudaGetErrorString(_err), __LINE__); \
            exit(EXIT_FAILURE);                      \
        }                                            \
    } while(0)


// =======================================================
// 2. TIMER SDK REPLACEMENT (StopWatchInterface)
// =======================================================

typedef struct
{
    cudaEvent_t start;
    cudaEvent_t stop;
} StopWatchInterface;


// create timer
inline void sdkCreateTimer(StopWatchInterface **timer)
{
    if (!timer) return;

    *timer = (StopWatchInterface*)malloc(sizeof(StopWatchInterface));

    if (!(*timer)) return;

    cudaEventCreate(&((*timer)->start));
    cudaEventCreate(&((*timer)->stop));
}


// destroy timer (important to avoid leaks)
inline void sdkDeleteTimer(StopWatchInterface *timer)
{
    if (!timer) return;

    cudaEventDestroy(timer->start);
    cudaEventDestroy(timer->stop);

    free(timer);
}


// start timer
inline void sdkStartTimer(StopWatchInterface *timer)
{
    cudaEventRecord(timer->start, 0);
}


// stop timer
inline void sdkStopTimer(StopWatchInterface *timer)
{
    cudaEventRecord(timer->stop, 0);
    cudaEventSynchronize(timer->stop);
}


// return elapsed time in ms
inline float sdkGetAverageTimerValue(StopWatchInterface *timer)
{
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, timer->start, timer->stop);
    return ms;
}


// legacy alias (some Rodinia versions use this name)
#define sdkResetTimer(timer) sdkStartTimer(timer)


// =======================================================
// 3. OPTIONAL SAFETY UTILITIES
// =======================================================

// simple device sync check (useful in debug builds)
inline void cudaSafeSync(const char *msg)
{
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA Sync Error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
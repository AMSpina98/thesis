#pragma once

#if defined(_WIN32) && !defined(__CUDACC__)

#include <windows.h>
#include <stdint.h>

struct timezone_win {
    int tz_minuteswest;
    int tz_dsttime;
};

static inline int gettimeofday(struct timeval *tp, void *tzp)
{
    (void)tzp;

    static const uint64_t EPOCH = 116444736000000000ULL;

    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);

    uint64_t t =
        ((uint64_t)ft.dwHighDateTime << 32) |
        (uint64_t)ft.dwLowDateTime;

    t -= EPOCH;

    tp->tv_sec  = (long)(t / 10000000ULL);
    tp->tv_usec = (long)((t % 10000000ULL) / 10);

    return 0;
}

#elif !defined(_WIN32)

#include <sys/time.h>

#endif
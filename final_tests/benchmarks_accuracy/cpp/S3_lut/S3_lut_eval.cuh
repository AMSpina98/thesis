// S3_lut_eval.cuh -- valutatore delle correzioni R+(k), R-(k) dell'addizionatore S3.
// Stessa aritmetica intera bit-exact del riferimento Python (gen_s3_tables.py, eval_int).
//
//   delta<F, true >(k) = R+(k) = round(S * ln(1 + e^(-k/S)))     k >= 0
//   delta<F, false>(k) = R-(k) = round(S * ln(1 - e^(-k/S)))     k >= 1   (k = 0 NON valido)
//   delta_sel<F>(k, same_sign)  = una delle due, scelta senza ramo (nessuna divergenza di warp)
//
// con k = mantissa maggiore - mantissa minore (>= 0) e S = 2^F.
//
// Struttura (ottimizzata per ridurre le istruzioni intere e le letture):
//  * k viene LIMITATO a K_CUT (una sola IMNMX): l'ultima voce di ogni tabella e' a zero, quindi
//    per k >= K_CUT il risultato e' 0 senza confronto + select.
//  * M e' comune a f+ e f-: la scelta f+/f- seleziona solo EMIN (se diverso) e l'offset di tabella.
//  * una sola lettura vettoriale per voce (8 B per ORDER=1, 16 B per ORDER=2).
//
// Larghezza: se i massimi calcolati dal generatore (VMAX, PMAX) lo permettono si lavora a
// 32 bit, altrimenti a 64. Gli shift a destra di interi negativi sono aritmetici
// (gcc/clang/nvcc/MSVC); le moltiplicazioni per 2^e evitano lo shift a sinistra di negativi.
#pragma once
#include <type_traits>
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#include <intrin.h>
#endif
#include "S3_lut_common.cuh"

namespace s3lut {

S3_HD inline uint32_t ilog2_u32(uint32_t x) {            // x >= 1
#ifdef __CUDA_ARCH__
    return 31u - (uint32_t)__clz((int)x);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanReverse(&idx, (unsigned long)x);                 // MSVC non ha __builtin_clz
    return (uint32_t)idx;
#else
    return 31u - (uint32_t)__builtin_clz(x);
#endif
}

// off = 0 per f+, OFF_MINUS per f-; emin = EMIN_PLUS / EMIN_MINUS. Entrambi arrivano gia' scelti.
template <int F>
S3_HD inline int32_t delta_core(uint32_t k, int emin, uint32_t off) {
    using T = Tab<F>;
    constexpr bool W32 = (T::VMAX < (1LL << 30)) && (T::PMAX < (1LL << 30));
    using W = typename std::conditional<W32, int32_t, int64_t>::type;
    const uint32_t kk = (k < T::K_CUT) ? k : T::K_CUT;        // IMNMX.U32: k >= K_CUT -> voce a zero
    int e = (int)ilog2_u32(kk | 1u) - T::M;
    e = (e > emin) ? e : emin;
    const uint32_t idx = off + ((uint32_t)(e - emin) << T::M) + (kk >> e);
    const typename T::Entry en = T::table()[idx];            // una sola lettura vettoriale
    const W r = (W)(kk & ((1u << e) - 1u));
    if constexpr (T::ORDER == 1) {
        const W V = (W)en.c0 * ((W)1 << e) + (W)en.c1 * r;
        const int sh = e + T::G;
        return (int32_t)((V + ((W)1 << (sh - 1))) >> sh);
    } else {
        const W half = ((W)1 << e) >> 1;
        W t = ((W)en.c2 * r + half) >> e;
        t = ((t + (W)en.c1) * r + half) >> e;
        const W V = (W)en.c0 + t;
        return (int32_t)((V + ((W)1 << (T::G - 1))) >> T::G);
    }
}

template <int F, bool PLUS>
S3_HD inline int32_t delta(uint32_t k) {
    using T = Tab<F>;
    return delta_core<F>(k, PLUS ? T::EMIN_PLUS : T::EMIN_MINUS, PLUS ? 0u : T::OFF_MINUS);
}

// Scelta f+/f- senza ramo: un select per l'offset e, solo se EMIN differisce, uno per EMIN.
template <int F>
S3_HD inline int32_t delta_sel(uint32_t k, bool same_sign) {
    using T = Tab<F>;
    const int emin = (T::EMIN_PLUS == T::EMIN_MINUS) ? T::EMIN_PLUS
                                                     : (same_sign ? T::EMIN_PLUS : T::EMIN_MINUS);
    const uint32_t off = same_sign ? 0u : T::OFF_MINUS;
    return delta_core<F>(k, emin, off);
}

} // namespace s3lut

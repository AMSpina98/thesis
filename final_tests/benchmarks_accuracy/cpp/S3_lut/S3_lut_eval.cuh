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

// Parte bassa a 32 bit di (p >> e) per p a 64 bit con segno e 0 <= e < 32: su GPU e' UNA sola SHF
// (funnel shift) al posto dello shift a 64 bit (2 istruzioni). Valida se il risultato sta in 32 bit.
S3_HD inline int32_t sra64_lo(int64_t p, int e) {
#ifdef __CUDA_ARCH__
    return (int32_t)__funnelshift_r((unsigned)p, (unsigned)((uint64_t)p >> 32), (unsigned)e);
#else
    return (int32_t)(p >> e);
#endif
}

// Valutazione del polinomio sul segmento scelto. Specializzazioni (invece di "if constexpr", cosi' il
// codice compila anche in C++14, es. nvcc 10.2 su Jetson Nano):
//   Eval<T,1,*>     : interpolazione lineare;
//   Eval<T,2,true>  : quadratico con UN solo prodotto a 64 bit (caso 32_21);
//   Eval<T,2,false> : quadratico generico (32 bit se VMAX e PMAX < 2^30, altrimenti 64).
// Le tre forme danno lo STESSO risultato (stessa formula, stessi arrotondamenti).
template <class T, int ORDER, bool MIX> struct Eval;

template <class T, bool MIX> struct Eval<T, 1, MIX> {
    static constexpr bool W32 = (T::VMAX < (1LL << 30)) && (T::PMAX < (1LL << 30));
    using W = typename std::conditional<W32, int32_t, int64_t>::type;
    S3_HD static int32_t run(const typename T::Entry& en, uint32_t kk, int e) {
        const W r = (W)(kk & ((1u << e) - 1u));
        const W V = (W)en.c0 * ((W)1 << e) + (W)en.c1 * r;
        const int sh = e + T::G;
        return (int32_t)((V + ((W)1 << (sh - 1))) >> sh);
    }
};

template <class T> struct Eval<T, 2, true> {
    S3_HD static int32_t run(const typename T::Entry& en, uint32_t kk, int e) {
        const int32_t r    = (int32_t)(kk & ((1u << e) - 1u));       // < 2^e <= 2^25
        const int32_t half = (int32_t)((1u << e) >> 1);
        const int32_t t    = (en.c2 * r + half) >> e;                  // 32 bit
        const int64_t p    = (int64_t)(t + en.c1) * (int64_t)r + (int64_t)half;   // IMAD.WIDE
        const int32_t V    = en.c0 + sra64_lo(p, e);
        return (V + (1 << (T::G - 1))) >> T::G;
    }
};

template <class T> struct Eval<T, 2, false> {
    static constexpr bool W32 = (T::VMAX < (1LL << 30)) && (T::PMAX < (1LL << 30));
    using W = typename std::conditional<W32, int32_t, int64_t>::type;
    S3_HD static int32_t run(const typename T::Entry& en, uint32_t kk, int e) {
        const W r = (W)(kk & ((1u << e) - 1u));
        const W half = ((W)1 << e) >> 1;
        W t = ((W)en.c2 * r + half) >> e;
        t = ((t + (W)en.c1) * r + half) >> e;
        const W V = (W)en.c0 + t;
        return (int32_t)((V + ((W)1 << (T::G - 1))) >> T::G);
    }
};

// off = 0 per f+, OFF_MINUS per f-; emin = EMIN_PLUS / EMIN_MINUS. Entrambi arrivano gia' scelti.
// Larghezze scelte a compile time dai massimi calcolati e verificati dal generatore.
template <int F>
S3_HD inline int32_t delta_core(uint32_t k, int emin, uint32_t off) {
    typedef Tab<F> T;
    constexpr bool W32 = (T::VMAX < (1LL << 30)) && (T::PMAX < (1LL << 30));
    constexpr bool MIX = !W32 && T::ORDER == 2 && T::VMAX < (1LL << 30) &&
                         T::P1MAX < (1LL << 30) && T::SMAX < (1LL << 30);
    const uint32_t kc = (uint32_t)T::K_CUT;
    const uint32_t kk = (k < kc) ? k : kc;                    // IMNMX.U32: k >= K_CUT -> voce a zero
    int e = (int)ilog2_u32(kk | 1u) - (int)T::M;
    e = (e > emin) ? e : emin;
    const uint32_t idx = off + ((uint32_t)(e - emin) << (int)T::M) + (kk >> e);
    const typename T::Entry en = T::table()[idx];            // una sola lettura vettoriale
    return Eval<T, (int)T::ORDER, MIX>::run(en, kk, e);
}

template <int F, bool PLUS>
S3_HD inline int32_t delta(uint32_t k) {
    using T = Tab<F>;
    return delta_core<F>(k, PLUS ? (int)T::EMIN_PLUS : (int)T::EMIN_MINUS, PLUS ? 0u : (uint32_t)T::OFF_MINUS);
}

// Scelta f+/f- senza ramo: un select per l'offset e, solo se EMIN differisce, uno per EMIN.
template <int F>
S3_HD inline int32_t delta_sel(uint32_t k, bool same_sign) {
    using T = Tab<F>;
    const int emin = (T::EMIN_PLUS == T::EMIN_MINUS) ? (int)T::EMIN_PLUS
                                                     : (same_sign ? (int)T::EMIN_PLUS : (int)T::EMIN_MINUS);
    const uint32_t off = same_sign ? 0u : (uint32_t)T::OFF_MINUS;
    return delta_core<F>(k, emin, off);
}

} // namespace s3lut

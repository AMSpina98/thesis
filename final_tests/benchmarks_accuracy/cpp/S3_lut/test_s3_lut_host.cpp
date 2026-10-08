// test_s3_lut_host.cpp -- verifica HOST (senza GPU) del valutatore S3 per tutti i formati generati.
//   g++ -std=c++17 -O2 -I. test_s3_lut_host.cpp -o test_s3_lut_host && ./test_s3_lut_host
// 1) checksum esaustivi (somma R, somma k*R, mod 2^64) identici a quelli del generatore Python
// 2) confronto indipendente con il valore ideale in long double (tutti i k per F<=12, campione oltre)
// 3) bordi: k=0 (somma), k=K_CUT-1, K_CUT, 0xFFFFFFFF
#include <cmath>
#include <cstdio>
#include "S3_tables_16_10.cuh"
#include "S3_tables_32_21.cuh"
#include "S3_lut_eval.cuh"

using namespace s3lut;

template <int F>
static int test() {
    using T = Tab<F>;
    int fail = 0;
    const long double S = std::ldexp(1.0L, F);
    uint64_t sp1 = 0, sp2 = 0, sm1 = 0, sm2 = 0;
    long double maxp = 0, maxm = 0;
    const uint32_t step = (F <= 12) ? 1u : 4099u;          // oltre F=12 controllo ideale su un campione
    for (uint32_t k = 1; k < T::K_CUT; ++k) {
        const int32_t rp = delta<F, true >(k);
        const int32_t rm = delta<F, false>(k);
        sp1 += (uint64_t)(int64_t)rp;  sp2 += (uint64_t)k * (uint64_t)(int64_t)rp;
        sm1 += (uint64_t)(int64_t)rm;  sm2 += (uint64_t)k * (uint64_t)(int64_t)rm;
        if ((k - 1) % step == 0) {
            const long double d = (long double)k / S;
            const long double ip = S * log1pl(expl(-d));
            const long double im = S * logl(-expm1l(-d));
            const long double ep = fabsl((long double)rp - ip), em = fabsl((long double)rm - im);
            if (ep > maxp) maxp = ep;
            if (em > maxm) maxm = em;
        }
    }
    auto chk = [&](const char* name, uint64_t got, uint64_t want) {
        if (got != want) { std::printf("  FAIL %-14s got %016llx want %016llx\n", name,
                                       (unsigned long long)got, (unsigned long long)want); ++fail; }
        else             { std::printf("  ok   %-14s %016llx\n", name, (unsigned long long)got); }
    };
    std::printf("F=%d  K_CUT=%u  ORDER=%d\n", F, T::K_CUT, T::ORDER);
    chk("checksum f+ #1", sp1, T::CHK_PLUS_1);  chk("checksum f+ #2", sp2, T::CHK_PLUS_2);
    chk("checksum f- #1", sm1, T::CHK_MINUS_1); chk("checksum f- #2", sm2, T::CHK_MINUS_2);
    std::printf("  |R - ideale| max (long double): f+ %.4Lf   f- %.4Lf  (arrotondamento perfetto = 0.5)\n", maxp, maxm);
    if (maxp > 0.75L || maxm > 0.75L) { std::printf("  FAIL errore rispetto all'ideale troppo alto\n"); ++fail; }
    if (delta<F, true>(0) != T::R0_PLUS) { std::printf("  FAIL R+(0)\n"); ++fail; }
    if (delta<F, true>(T::K_CUT) != 0 || delta<F, false>(T::K_CUT) != 0 ||
        delta<F, true>(0xFFFFFFFFu) != 0 || delta<F, false>(0xFFFFFFFFu) != 0) { std::printf("  FAIL bordo K_CUT\n"); ++fail; }
    if (!fail) std::printf("  tutto ok\n");
    return fail;
}

int main() {
    int fail = 0;
    fail += test<10>();
    fail += test<21>();
    std::printf(fail ? "\nRISULTATO: %d FALLIMENTI\n" : "\nRISULTATO: TUTTO OK\n", fail);
    return fail ? 1 : 0;
}

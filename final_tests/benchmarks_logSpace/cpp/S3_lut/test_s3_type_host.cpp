// test_s3_type_host.cpp -- test HOST (g++, senza GPU) del tipo LNNS_S3 contro:
//   (a) il vero S2.cuh (LNNS) per casi speciali, *, /, confronti (devono essere identici bit a bit);
//   (b) il valore ideale in long double per + e - (errore sulla mantissa <= 0,75 LSB);
//   (c) conversioni di bordo e saturazioni.
// Build (S2.cuh e' nella cartella superiore; stub/ fornisce un cuda_fp16.h vuoto):
//   g++ -std=c++17 -O2 -Wall -Wextra -I. -Istub -I.. test_s3_type_host.cpp -o test_s3_type_host && ./test_s3_type_host
#define __host__
#define __device__
#include "S2.cuh"
#include "S3.cuh"
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; if (g_fail < 40) { std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } } while (0)

template <typename T, int F>
static void run(const char* name, uint64_t seed) {
    using S2 = LNNS<T, F>;
    using S3 = LNNS_S3<T, F>;
    using U = T;
    static_assert(sizeof(S2) == sizeof(S3), "size");
    constexpr int MB = S3::MANTISSA_BITS;
    constexpr long double SC = (long double)((uint64_t)1 << F);
    const int fail0 = g_fail;
    std::printf("== %s ==\n", name);

    auto unp = [](U v) { return (int32_t)((uint32_t)v << (32 - MB)) >> (32 - MB); };
    auto mk = [&](bool neg, int32_t m) { return (U)((neg ? S3::SIGN_MASK : 0) | ((U)(uint32_t)m & S3::MANTISSA_MASK)); };

    // ---- (a) griglia di valori speciali + casuali: tutte le coppie ----
    std::vector<U> v = { S3::LOG_ZERO, S3::LOG_NAN, mk(0, 0), mk(1, 0), mk(0, 1), mk(1, 1), mk(0, S3::MAX_MANTISSA),
                         mk(1, S3::MAX_MANTISSA), mk(0, S3::MIN_MANTISSA), mk(1, S3::MIN_MANTISSA),
                         mk(0, 1 << F), mk(1, 1 << F), mk(0, -(1 << F)), mk(1, 3 << (F - 1)) };
    std::mt19937_64 rng(seed);
    for (int i = 0; i < 40; ++i) v.push_back((U)rng());
    long long maxd = 0, nadd = 0, nsame = 0;
    for (U a : v) for (U b : v) {
        S2 a2 = S2::raw(a), b2 = S2::raw(b);
        S3 a3 = S3::raw(a), b3 = S3::raw(b);
        CHECK(((a2 * b2).value) == ((a3 * b3).value), "mul %x %x", (unsigned)a, (unsigned)b);
        CHECK(((a2 / b2).value) == ((a3 / b3).value), "div %x %x", (unsigned)a, (unsigned)b);
        CHECK((a2 < b2) == (a3 < b3) && (a2 <= b2) == (a3 <= b3) && (a2 > b2) == (a3 > b3) &&
              (a2 >= b2) == (a3 >= b3) && (a2 == b2) == (a3 == b3) && (a2 != b2) == (a3 != b3), "cmp %x %x", (unsigned)a, (unsigned)b);
        { S2 t2 = a2; t2 *= b2; S3 t3 = a3; t3 *= b3; CHECK(t2.value == t3.value, "mul= %x %x", (unsigned)a, (unsigned)b); }
        { S2 t2 = a2; t2 /= b2; S3 t3 = a3; t3 /= b3; CHECK(t2.value == t3.value, "div= %x %x", (unsigned)a, (unsigned)b); }
        for (int sub = 0; sub < 2; ++sub) {
            S2 r2 = sub ? a2 - b2 : a2 + b2;
            S3 r3 = sub ? a3 - b3 : a3 + b3;
            S3 q3 = a3; if (sub) q3 -= b3; else q3 += b3;
            CHECK(q3.value == r3.value, "op= != op");
            const bool special = a == S3::LOG_NAN || b == S3::LOG_NAN || a == S3::LOG_ZERO || b == S3::LOG_ZERO;
            if (special) { CHECK(r2.value == r3.value, "special add/sub %d %x %x: S2 %x S3 %x", sub, (unsigned)a, (unsigned)b, (unsigned)r2.value, (unsigned)r3.value); continue; }
            if (r2.value == S3::LOG_ZERO || r3.value == S3::LOG_ZERO) { CHECK(r2.value == r3.value, "zero mismatch %x %x", (unsigned)a, (unsigned)b); continue; }
            ++nadd;
            const bool sgn_eq = (r2.value & S3::SIGN_MASK) == (r3.value & S3::SIGN_MASK);
            const long long d = std::llabs((long long)unp(r2.value) - (long long)unp(r3.value));
            if (sgn_eq) { ++nsame; if (d > maxd) maxd = d; }
            else CHECK(false, "segno diverso %d %x %x", sub, (unsigned)a, (unsigned)b);
        }
    }
    std::printf("  speciali/mul/div/confronti vs S2: ok; add/sub: %lld coppie, max |S3-S2| = %lld LSB (S2 ha errore float)\n", nsame, maxd);
    (void)nadd;

    // ---- (b) coppie casuali contro l'ideale in long double ----
    long double worst = 0; long long n = 0;
    std::uniform_int_distribution<int32_t> dm(S3::MIN_MANTISSA, S3::MAX_MANTISSA);
    std::uniform_int_distribution<int32_t> dsmall(-(40 << F) , 40 << F);
    std::uniform_int_distribution<int> dk(0, 3);
    for (long long it = 0; it < 3000000; ++it) {
        int32_t mA, mB;
        switch (dk(rng)) {
            case 0: mA = dm(rng); mB = dm(rng); break;
            case 1: mA = dsmall(rng); mB = mA + (int32_t)(rng() % (12u << F)) - (6 << F); break;
            case 2: mA = dsmall(rng); mB = mA + (int32_t)(rng() % 64) - 32; break;
            default: mA = dsmall(rng); mB = dsmall(rng); break;
        }
        mA = std::max(S3::MIN_MANTISSA, std::min(S3::MAX_MANTISSA, mA));
        mB = std::max(S3::MIN_MANTISSA, std::min(S3::MAX_MANTISSA, mB));
        const bool na = rng() & 1, nb = rng() & 1, sub = rng() & 1;
        S3 a = S3::raw(mk(na, mA)), b = S3::raw(mk(nb, mB));
        S3 r = sub ? a - b : a + b;
        const bool eff_neg_b = nb ^ sub;
        if (mA == mB && na != eff_neg_b) { CHECK(r.value == S3::LOG_ZERO, "cancellazione"); continue; }
        const int32_t mM = std::max(mA, mB), mm = std::min(mA, mB);
        const bool sM = (mB > mA) ? eff_neg_b : na;
        const bool same = (na == eff_neg_b);
        const long double d = (long double)((int64_t)mM - mm) / SC;
        const long double ideal = (long double)mM + SC * (same ? log1pl(expl(-d)) : logl(-expm1l(-d)));
        const int32_t rm = unp(r.value);
        CHECK(((r.value & S3::SIGN_MASK) != 0) == sM, "segno");
        if (ideal >= (long double)S3::MAX_MANTISSA) { CHECK(rm == S3::MAX_MANTISSA, "clamp alto"); continue; }
        if (ideal <= (long double)S3::MIN_MANTISSA) { CHECK(rm == S3::MIN_MANTISSA, "clamp basso"); continue; }
        const long double e = fabsl((long double)rm - ideal);
        if (e > worst) worst = e;
        CHECK(e <= 0.75L, "errore %.3Lf  mA=%d mB=%d na=%d nb=%d sub=%d", e, mA, mB, (int)na, (int)nb, (int)sub);
        ++n;
    }
    std::printf("  %lld coppie casuali vs ideale: errore massimo sulla mantissa = %.4Lf LSB (limite 0,75)\n", n, worst);

    // ---- (c) conversioni ----
    CHECK(S3(0.0).value == S3::LOG_ZERO && S3(0.0f).value == S3::LOG_ZERO && S3(0).value == S3::LOG_ZERO, "ctor 0");
    CHECK(S3(std::numeric_limits<double>::quiet_NaN()).value == S3::LOG_NAN, "ctor nan");
    CHECK(S3(std::numeric_limits<float>::quiet_NaN()).value == S3::LOG_NAN, "ctor nanf");
    CHECK(unp(S3(1.0).value) == 0 && S3(1.0).value == mk(0, 0), "ctor 1");
    CHECK(S3(-1.0).value == mk(1, 0), "ctor -1");
    CHECK(unp(S3(1e300).value) == S3::MAX_MANTISSA || F == 21, "sat alta double");
    CHECK(unp(S3(std::numeric_limits<double>::infinity()).value) == S3::MAX_MANTISSA, "sat inf");
    CHECK(unp(S3(1e-300).value) == S3::MIN_MANTISSA || F == 21, "sat bassa double");
    CHECK(S3(1e-320).value != S3::LOG_ZERO, "denormale non zero");
    for (double x : {0.37, 1.5, 2.0, 3.14159, 100.0, 12345.0}) {
        const double rt = S3(x).toDouble(), rt2 = S3(-x).toDouble(), rf = S3((float)x).toFloat();
        const double tol = (F == 10) ? 5e-3 : 5e-6;
        CHECK(std::fabs(rt / x - 1) < tol && std::fabs(rt2 / -x - 1) < tol && std::fabs(rf / x - 1) < tol, "roundtrip %g -> %g %g %g", x, rt, rt2, rf);
    }
    // toInt/toUInt troncano (come S2): 6,9999 -> 6, quindi si accetta uno scarto di 1
    CHECK(std::abs(S3(7).toInt() - 7) <= 1 && std::abs(S3(-7).toInt() + 7) <= 1 && (int)S3(9u).toUInt() >= 8 && S3(9u).toUInt() <= 9u, "int roundtrip %d", S3(7).toInt());
    if (F == 21) {                                   // il punto di S3: range da double
        for (double x : {1e50, 1e100, 1e200, 1e-100, 1e-200}) {
            const double rt = S3(x).toDouble();
            CHECK(std::fabs(rt / x - 1) < 5e-6, "range double %g -> %g", x, rt);
        }
        S3 big(1e200), one(1.0);
        CHECK(unp((big * big * big).value) == S3::MAX_MANTISSA, "saturazione a e^512 (~1e222)");
        CHECK(std::fabs((big + one).toDouble() / 1e200 - 1) < 5e-6, "1e200+1");
        CHECK(std::fabs((big - big).toDouble()) == 0.0, "x-x");
        CHECK(std::fabs((S3(3.0) + S3(5.0)).toDouble() - 8.0) < 8.0 * 5e-6, "3+5");
        CHECK(std::fabs((S3(3.0) - S3(5.0)).toDouble() + 2.0) < 2.0 * 5e-6, "3-5");
    } else {
        CHECK(std::fabs((S3(3.0f) + S3(5.0f)).toFloat() - 8.0f) < 8.0f * 5e-3f, "3+5 (16)");
        CHECK(std::fabs((S3(3.0f) - S3(5.0f)).toFloat() + 2.0f) < 2.0f * 5e-3f, "3-5 (16)");
    }
    std::printf("  %s\n", g_fail == fail0 ? "OK" : "FALLITO");
}

int main() {
    run<uint16_t, 10>("S3_16_10 (lineare)", 1);
    run<uint32_t, 21>("S3_32_21 (quadratico)", 2);
    std::printf("\nRISULTATO: %s (%d fallimenti)\n", g_fail ? "FALLITO" : "TUTTO OK", g_fail);
    return g_fail ? 1 : 0;
}

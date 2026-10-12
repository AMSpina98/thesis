// S3.cuh -- LNNS_S3: stesso formato e stessa interfaccia di LNNS (S2.cuh), ma la somma/sottrazione
// usa la tabella intera (s3lut::delta_sel) invece di Log/Exp in virgola mobile.
// Formati supportati (quelli per cui esistono le tabelle):
//     S3_16_10 = LNNS_S3<uint16_t,10>   (interpolazione lineare,   ~2,5 KB)
//     S3_32_21 = LNNS_S3<uint32_t,21>   (interpolazione quadratica, ~17,9 KB)
// Autonomo: NON include S2.cuh (si possono includere entrambi, i nomi non collidono).
#pragma once

#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>
#include "S3_lut_common.cuh"
#include "S3_tables_16_10.cuh"
#include "S3_tables_32_21.cuh"
#include "S3_lut_eval.cuh"

namespace s3math {
    // Solo per le conversioni di bordo (da/verso float, double, int): fuori dal cammino caldo di + e -.
    S3_HD inline double Log(double x) {
    #ifdef __CUDA_ARCH__
        return ::log(x);
    #else
        return std::log(x);
    #endif
    }
    S3_HD inline double Exp(double x) {
    #ifdef __CUDA_ARCH__
        return ::exp(x);
    #else
        return std::exp(x);
    #endif
    }
    S3_HD inline float LogF(float x) {
    #ifdef __CUDA_ARCH__
        return __logf(x);
    #else
        return std::log(x);
    #endif
    }
    S3_HD inline float ExpF(float x) {
    #ifdef __CUDA_ARCH__
        return __expf(x);
    #else
        return std::exp(x);
    #endif
    }
    // double (gia' limitato a +-2^30) -> int32 arrotondato al piu' vicino (pari sui .5)
    S3_HD inline int32_t RoundToI32(double x) {
    #ifdef __CUDA_ARCH__
        return __double2int_rn(x);
    #else
        return (int32_t)std::nearbyint(x);
    #endif
    }
    S3_HD inline bool IsNaN(double x) { return x != x; }
    S3_HD inline double Abs(double x) { return x < 0.0 ? -x : x; }
    S3_HD inline double QNaN() {
    #ifdef __CUDA_ARCH__
        return __longlong_as_double(0x7ff8000000000000LL);
    #else
        return std::numeric_limits<double>::quiet_NaN();
    #endif
    }
}

template <typename T_, int PRECISION_>
struct LNNS_S3 {
    using StorageType = T_;
    using SignedStorageType = std::make_signed_t<StorageType>;

    static constexpr int STORAGE_BITS  = (int)sizeof(StorageType) * 8;
    static constexpr int FRACTIONAL_BITS = PRECISION_;
    static constexpr int MANTISSA_BITS = STORAGE_BITS - 1;                 // bit di segno escluso
    static constexpr int INT_BITS      = MANTISSA_BITS - FRACTIONAL_BITS;
    static constexpr double SCALE_FACTOR = (double)((uint64_t)1 << FRACTIONAL_BITS);
    static constexpr StorageType SIGN_MASK          = (StorageType)((uint32_t)1 << MANTISSA_BITS);
    static constexpr StorageType MANTISSA_SIGN_MASK = (StorageType)((uint32_t)1 << (MANTISSA_BITS - 1));
    static constexpr StorageType MANTISSA_MASK      = (StorageType)(((uint32_t)1 << MANTISSA_BITS) - 1u);
    static constexpr int32_t MAX_MANTISSA = (int32_t)(((uint32_t)1 << (MANTISSA_BITS - 1)) - 1u);
    static constexpr int32_t MIN_MANTISSA = -MAX_MANTISSA;
    static constexpr StorageType LOG_ZERO = MANTISSA_SIGN_MASK;
    static constexpr StorageType LOG_NAN  = (StorageType)(SIGN_MASK | MANTISSA_SIGN_MASK);

    // ---- politiche di costo (scelte a compile time per formato) --------------------------------
    // SATURATE: clamp della mantissa a +-MAX_MANTISSA dopo + - * /. Senza clamp un overflow del campo
    //   (|ln x| >= 2^(MANTISSA_BITS-1)/2^F) NON satura: il campo si avvolge. A 32 bit il campo copre ln +-512
    //   (e^512 ~ 1e222, ben oltre double/float) quindi si omette; a 16 bit (ln +-16) il clamp resta.
    //   Forzabile con -DS3_ALWAYS_SATURATE.
    // ZERO_VIA_K: lo zero (e il NaN) hanno mantissa MINM = -2^(MANTISSA_BITS-1), inferiore a ogni valore
    //   normale: in una somma k = mMax - mMin e' enorme (>= K_CUT => R = 0) e il percorso normale restituisce
    //   gia' l'altro operando, senza select dedicate. Esatto per ogni operando con |ln| < 2^(MANTISSA_BITS-1)/2^F
    //   - K_CUT/2^F (a 32 bit: x > e^-496,7 ~ 1e-216); a 16 bit K_CUT e' troppo grande (>= e^-8,4 ~ 2e-4)
    //   quindi li' le select restano. Forzabile con -DS3_EXACT_SPECIALS.
#ifdef S3_ALWAYS_SATURATE
    static constexpr bool SATURATE = true;
#else
    static constexpr bool SATURATE = (sizeof(StorageType) == 2);
#endif
#ifdef S3_EXACT_SPECIALS
    static constexpr bool ZERO_VIA_K = false;
#else
    static constexpr bool ZERO_VIA_K = (sizeof(StorageType) == 4);
#endif
    static constexpr int32_t MINM = -(int32_t)((uint32_t)1 << (MANTISSA_BITS - 1));   // mantissa di LOG_ZERO / LOG_NAN

    static_assert((sizeof(StorageType) == 2 && PRECISION_ == 10) ||
                  (sizeof(StorageType) == 4 && PRECISION_ == 21),
                  "LNNS_S3: tabelle disponibili solo per <uint16_t,10> e <uint32_t,21> (altri formati: gen_s3_tables.py)");

    StorageType value;

  private:
    struct RawTag {};
    S3_HD LNNS_S3(StorageType v, RawTag) : value(v) {}

    // ---- mantissa con segno, estratta una volta sola (SHL + SAR) ----
    S3_HD static int32_t unpack(StorageType v) {
        constexpr int sh = 32 - MANTISSA_BITS;
        return (int32_t)((uint32_t)v << sh) >> sh;   // i bit sopra MANTISSA_BITS (segno del numero) escono con lo shift
    }
    // (tag-dispatch al posto di "if constexpr": compila anche in C++14; i cast evitano l'odr-use dei constexpr)
    S3_HD static int32_t clampImpl(int32_t m, std::false_type) { return m; }
    S3_HD static int32_t clampImpl(int32_t m, std::true_type) {
        return m > (int32_t)MAX_MANTISSA ? (int32_t)MAX_MANTISSA : (m < (int32_t)MIN_MANTISSA ? (int32_t)MIN_MANTISSA : m);
    }
    S3_HD static int32_t clampM(int32_t m) { return clampImpl(m, std::integral_constant<bool, SATURATE>()); }
    S3_HD static StorageType pack(StorageType sign, int32_t m) {
        return (StorageType)(sign | ((StorageType)(uint32_t)m & MANTISSA_MASK));
    }

    // ---- nuclei SENZA RAMI -------------------------------------------------------------------
    // Il percorso normale viene calcolato sempre (e' sicuro anche con NaN/zero: k e' limitato a K_CUT,
    // gli indici restano in tabella, nessun overflow) e i casi speciali sovrascrivono il risultato con
    // select, in ordine di priorita' crescente. Niente return anticipati => niente BSSY/BSYNC/BMOV.
    // (In una sottrazione "b" e' gia' stato invertito di segno nel calcolo di sB, mai nei confronti.)
    // zero/NaN come operando, con ZERO_VIA_K: il percorso normale ha gia' dato l'altro operando (k >= K_CUT, R = 0).
    // Restano x-x e (zero,zero), ossia mA == mB con segni opposti oppure entrambi speciali.
    template <bool SUB>
    S3_HD static StorageType zero_cases(StorageType res, StorageType, StorageType, int32_t mA, int32_t mB, bool same, std::true_type) {
        return (mA == mB && (!same || mA == MINM)) ? (StorageType)LOG_ZERO : res;
    }
    template <bool SUB>
    S3_HD static StorageType zero_cases(StorageType res, StorageType a, StorageType b, int32_t mA, int32_t mB, bool same, std::false_type) {
        res = (mA == mB && !same) ? (StorageType)LOG_ZERO : res;      // x - x
        res = (a == LOG_ZERO) ? (SUB ? (StorageType)(SIGN_MASK ^ b) : b) : res;
        res = (b == LOG_ZERO) ? a : res;                              // priorita' su a == 0 (anche in sub)
        return res;
    }

    template <bool SUB>
    S3_HD static StorageType add_core(StorageType a, StorageType b) {
        const int32_t mA = unpack(a), mB = unpack(b);
        const StorageType sA = (StorageType)(a & SIGN_MASK);
        const StorageType sB = SUB ? (StorageType)((b & SIGN_MASK) ^ SIGN_MASK) : (StorageType)(b & SIGN_MASK);
        const bool same = (sA == sB);
        const bool sw = (mB > mA);
        const int32_t mMax = sw ? mB : mA;
        const int32_t mMin = sw ? mA : mB;
        const StorageType sg = sw ? sB : sA;
        const uint32_t k = (uint32_t)mMax - (uint32_t)mMin;          // >= 0, < 2^32
        const int32_t R = s3lut::delta_sel<PRECISION_>(k, same);
        StorageType res = pack(sg, clampM(mMax + R));
        res = zero_cases<SUB>(res, a, b, mA, mB, same, std::integral_constant<bool, ZERO_VIA_K>());
        res = (a == LOG_NAN || b == LOG_NAN) ? (StorageType)LOG_NAN : res;
        return res;
    }

    S3_HD static StorageType mul_core(StorageType a, StorageType b) {
        StorageType res = pack((StorageType)((a ^ b) & SIGN_MASK), clampM(unpack(a) + unpack(b)));
        res = (a == LOG_ZERO || b == LOG_ZERO) ? (StorageType)LOG_ZERO : res;
        res = (a == LOG_NAN || b == LOG_NAN) ? (StorageType)LOG_NAN : res;
        return res;
    }
    S3_HD static StorageType div_core(StorageType a, StorageType b) {
        StorageType res = pack((StorageType)((a ^ b) & SIGN_MASK), clampM(unpack(a) - unpack(b)));
        res = (a == LOG_ZERO) ? (StorageType)LOG_ZERO : res;
        res = (a == LOG_NAN || b == LOG_NAN || b == LOG_ZERO) ? (StorageType)LOG_NAN : res;
        return res;
    }

    // mantissa scalata (gia' in unita' LSB, double) -> codice, con saturazione prima della conversione
    S3_HD static StorageType from_scaled(bool neg, double sc) {
        const double lim = (double)MAX_MANTISSA;
        sc = sc > lim ? lim : (sc < -lim ? -lim : sc);
        return pack(neg ? SIGN_MASK : (StorageType)0, s3math::RoundToI32(sc));
    }
    S3_HD static StorageType from_double(double x) {
        if (x == 0.0) return LOG_ZERO;
        if (s3math::IsNaN(x)) return LOG_NAN;
        return from_scaled(x < 0.0, s3math::Log(s3math::Abs(x)) * SCALE_FACTOR);
    }

  public:
    S3_HD static LNNS_S3 raw(StorageType v) { return LNNS_S3(v, RawTag{}); }

    LNNS_S3() = default;

    // 16 bit: range ln +-16, log in float (come S2). 32 bit: log in double (il range arriva a ln +-512).
    S3_HD LNNS_S3(float x) {
        if (x == 0.0f) { value = LOG_ZERO; return; }
        if (x != x)    { value = LOG_NAN;  return; }
        if (sizeof(StorageType) == 2) {
            const float y = s3math::LogF(x < 0.0f ? -x : x);
            value = from_scaled(x < 0.0f, (double)(y * (float)SCALE_FACTOR));
        } else {
            value = from_double((double)x);
        }
    }
    S3_HD LNNS_S3(double x) { value = from_double(x); }
    S3_HD LNNS_S3(int x)          { value = from_double((double)x); }
    S3_HD LNNS_S3(unsigned int x) { value = from_double((double)x); }

    S3_HD explicit operator StorageType() const { return value; }

    // ---- aritmetica ----
    S3_HD LNNS_S3 operator+(const LNNS_S3& o) const { return raw(add_core<false>(value, o.value)); }
    S3_HD LNNS_S3 operator-(const LNNS_S3& o) const { return raw(add_core<true >(value, o.value)); }
    S3_HD LNNS_S3& operator+=(const LNNS_S3& o) { value = add_core<false>(value, o.value); return *this; }
    S3_HD LNNS_S3& operator-=(const LNNS_S3& o) { value = add_core<true >(value, o.value); return *this; }
    S3_HD LNNS_S3 operator*(const LNNS_S3& o) const { return raw(mul_core(value, o.value)); }
    S3_HD LNNS_S3 operator/(const LNNS_S3& o) const { return raw(div_core(value, o.value)); }
    S3_HD LNNS_S3& operator*=(const LNNS_S3& o) { value = mul_core(value, o.value); return *this; }
    S3_HD LNNS_S3& operator/=(const LNNS_S3& o) { value = div_core(value, o.value); return *this; }

    // ---- confronti (stessa semantica di S2: NaN -> false, != -> true) ----
    S3_HD bool operator<(const LNNS_S3& o) const {
        if (value == LOG_NAN || o.value == LOG_NAN) return false;
        const StorageType sA = value & SIGN_MASK, sB = o.value & SIGN_MASK;
        if (sA != sB) return sA != 0;
        return sA == 0 ? unpack(value) < unpack(o.value) : unpack(value) > unpack(o.value);
    }
    S3_HD bool operator<=(const LNNS_S3& o) const {
        if (value == LOG_NAN || o.value == LOG_NAN) return false;
        const StorageType sA = value & SIGN_MASK, sB = o.value & SIGN_MASK;
        if (sA != sB) return sA != 0;
        return sA == 0 ? unpack(value) <= unpack(o.value) : unpack(value) >= unpack(o.value);
    }
    S3_HD bool operator>(const LNNS_S3& o) const {
        if (value == LOG_NAN || o.value == LOG_NAN) return false;
        const StorageType sA = value & SIGN_MASK, sB = o.value & SIGN_MASK;
        if (sA != sB) return sB != 0;
        return sA == 0 ? unpack(value) > unpack(o.value) : unpack(value) < unpack(o.value);
    }
    S3_HD bool operator>=(const LNNS_S3& o) const {
        if (value == LOG_NAN || o.value == LOG_NAN) return false;
        const StorageType sA = value & SIGN_MASK, sB = o.value & SIGN_MASK;
        if (sA != sB) return sB != 0;
        return sA == 0 ? unpack(value) >= unpack(o.value) : unpack(value) <= unpack(o.value);
    }
    S3_HD bool operator==(const LNNS_S3& o) const {
        if (value == LOG_NAN || o.value == LOG_NAN) return false;
        if ((value & SIGN_MASK) != (o.value & SIGN_MASK)) return false;
        return unpack(value) == unpack(o.value);
    }
    S3_HD bool operator!=(const LNNS_S3& o) const {
        if (value == LOG_NAN || o.value == LOG_NAN) return true;
        if ((value & SIGN_MASK) != (o.value & SIGN_MASK)) return true;
        return unpack(value) != unpack(o.value);
    }

    // ---- conversioni verso i tipi nativi ----
    // Le conversioni usano exp in double: l'intero range del formato (fino a e^512 per 32_21) resta rappresentabile.
    S3_HD double toDouble() const {
        if (value == LOG_ZERO) return 0.0;
        if (value == LOG_NAN) return s3math::QNaN();
        const double r = s3math::Exp((double)unpack(value) / SCALE_FACTOR);
        return (value & SIGN_MASK) ? -r : r;
    }
    S3_HD float toFloat() const {
        if (sizeof(StorageType) == 2) {                              // range ln +-16: float basta
            if (value == LOG_ZERO) return 0.0f;
            if (value == LOG_NAN) return (float)s3math::QNaN();
            const float r = s3math::ExpF((float)unpack(value) / (float)SCALE_FACTOR);
            return (value & SIGN_MASK) ? -r : r;
        }
        return (float)toDouble();                                    // fuori da +-3.4e38 -> +-inf / 0
    }
    S3_HD int toInt() const {
        if (value == LOG_ZERO || value == LOG_NAN) return 0;
        const double r = toDouble();
        if (r >= 2147483647.0)  return 2147483647;
        if (r <= -2147483648.0) return (int)(-2147483647 - 1);
        return (int)r;
    }
    S3_HD unsigned int toUInt() const {
        if (value == LOG_ZERO || value == LOG_NAN) return 0;
        const double r = toDouble();
        if (r >= 4294967295.0) return 4294967295u;
        if (r <= 0.0) return 0u;
        return (unsigned int)r;
    }

    template <typename U = StorageType, typename = typename std::enable_if<!std::is_same<U, float>::value>::type>
    S3_HD operator float() const { return toFloat(); }
    template <typename U = StorageType, typename = typename std::enable_if<!std::is_same<U, double>::value>::type>
    S3_HD operator double() const { return toDouble(); }
    template <typename U = StorageType, typename = typename std::enable_if<!std::is_same<U, int>::value>::type>
    S3_HD operator int() const { return toInt(); }
    template <typename U = StorageType, typename = typename std::enable_if<!std::is_same<U, unsigned int>::value>::type>
    S3_HD operator unsigned int() const { return toUInt(); }
};

using S3_16_10 = LNNS_S3<uint16_t, 10>;
using S3_32_21 = LNNS_S3<uint32_t, 21>;

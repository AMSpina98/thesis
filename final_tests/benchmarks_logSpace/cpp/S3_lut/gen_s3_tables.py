#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_s3_tables.py -- tabelle per la somma/sottrazione a tabella (S3) del sistema LNNS.

Idea. In LNNS la somma di due valori con mantissa (log fissa) mA >= mB e' esatta e intera
tranne una funzione di UN SOLO intero, k = mA - mB >= 0:

    somma         mC = mA + R+(k),   R+(k) = round( S * ln(1 + e^(-k/S)) )
    sottrazione   mC = mA + R-(k),   R-(k) = round( S * ln(1 - e^(-k/S)) )      (k >= 1)

con S = 2^F (F = bit frazionari). Oltre K_CUT entrambe valgono 0. Si tabulano R+ e R-.

Schema. Segmenti "per ottava": detto j = floor(log2 k), e = max(j - M, EMIN),
    idx = ((e - EMIN) << M) + (k >> e),     r = k & (2^e - 1)
La tabella ha un segmento per idx, di larghezza 2^e. Con EMIN=0 i primi segmenti hanno
larghezza 1 (valori esatti) e poi 2^M segmenti per ottava: serve a f- (singolare in 0).
f+ e' liscia: usa EMIN>0 (larghezza minima costante) e meno segmenti per ottava.
L'errore dell'interpolazione e' ~ costante su tutto il range.
  ORDER=1: T[idx] (nodi), V = (T[i] << e) + (T[i+1]-T[i]) * r,  R = (V + half) >> (e+G)
  ORDER=2: (c0,c1,c2) per segmento (Lagrange su t=0,1/2,1), con u = r / 2^e:
           t = (c2*r + 2^(e-1)) >> e;  t = ((t + c1)*r + 2^(e-1)) >> e;  V = c0 + t;
           R = (V + 2^(G-1)) >> G
G = bit di guardia delle voci. Tutta l'aritmetica e' intera: il generatore la simula bit a bit
(numpy int64) e verifica TUTTI i k in [1, K_CUT) contro il valore ideale (float64, controllato
con mpmath), poi emette l'header .cuh con parametri, errori e checksum.

Uso:   python3 gen_s3_tables.py --preset 16_10
       python3 gen_s3_tables.py --preset 32_21
       python3 gen_s3_tables.py --F 12 --order 1 --g 6 --target 0.05
"""
import argparse, hashlib, math, os, sys, time
import numpy as np
from mpmath import mp, mpf, log as mlog, exp as mexp, nint as mnint

mp.dps = 60

PRESETS = {  # share_emin: stesso EMIN per f+ e f- (una select in meno) - vantaggioso solo se la tabella f+ resta piccola
    "16_10": dict(F=10, order=1, g=6, target=0.05, share_emin=True),
    "32_21": dict(F=21, order=2, g=4, target=0.25, share_emin=False),
}

# ----------------------------------------------------------------------------------------
# funzioni ideali
# ----------------------------------------------------------------------------------------
def f_np(which, k, F):
    """valore ideale in unita' di LSB (float64). which: 'p' somma, 'm' sottrazione."""
    S = float(2 ** F)
    d = np.asarray(k, dtype=np.float64) / S
    if which == "p":
        return S * np.log1p(np.exp(-d))
    with np.errstate(divide="ignore"):
        return S * np.log(-np.expm1(-d))

def f_mp(which, k, F):
    S = mpf(2) ** F
    x = mexp(-mpf(k) / S)
    return S * (mlog(1 + x) if which == "p" else mlog(1 - x))

def compute_kcut(F):
    """minimo k oltre il quale R+ e R- arrotondano entrambe a 0 (f+ < 0.5 e |f-| < 0.5)."""
    lo, hi = 1, 1 << 40
    while lo < hi:
        mid = (lo + hi) // 2
        if f_np("p", mid, F) < 0.5 and -f_np("m", mid, F) < 0.5:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)

# ----------------------------------------------------------------------------------------
# segmentazione per ottava
# ----------------------------------------------------------------------------------------
def seg_index(k, m, emin=0):
    k = np.asarray(k, dtype=np.int64)
    j = (np.frexp(np.maximum(k, 1).astype(np.float64))[1] - 1).astype(np.int64)
    e = np.maximum(j - m, emin)
    idx = np.left_shift(e - emin, m) + np.right_shift(k, e)
    r = k & (np.left_shift(np.int64(1), e) - 1)
    return idx, e, r

def seg_start(idx, m, emin=0):
    idx = np.asarray(idx, dtype=np.int64)
    eo = np.where(idx < (2 << m), 0, np.right_shift(idx, m) - 1)
    e = eo + emin
    q = idx - np.left_shift(eo, m)
    return np.left_shift(q, e), e

def num_segments(kcut, m, emin=0):
    idx, _, _ = seg_index(np.array([kcut - 1]), m, emin)
    return int(idx[0]) + 1

ENTRY_W = {1: 2, 2: 4}          # int32 per voce: (T, dT) oppure (c0, c1, c2, pad): una sola lettura vettoriale

def table_bytes(order, nseg):
    return 4 * ENTRY_W[order] * (nseg + 1)

# ----------------------------------------------------------------------------------------
# costruzione tabelle (quantizzate)
# ----------------------------------------------------------------------------------------
def build_table(which, F, order, g, m, emin, kcut, hp=True):
    """hp=True: nodi con mpmath (tabella finale). hp=False: float64 vettoriale (ricerca)."""
    nseg = num_segments(kcut, m, emin)
    scale = mpf(2) ** g
    lo = 1 if which == "m" else 0
    if order == 1:
        starts, _ = seg_start(np.arange(nseg + 1), m, emin)
        if hp:
            T = np.array([0 if (s == 0 and which == "m") else int(mnint(f_mp(which, int(s), F) * scale))
                          for s in starts], dtype=np.int64)
        else:
            T = np.rint(f_np(which, np.maximum(starts, lo), F) * 2.0 ** g).astype(np.int64)
            if which == "m":
                T[starts == 0] = 0
        return T
    C = np.zeros((nseg, 3), dtype=np.int64)
    starts, es = seg_start(np.arange(nseg), m, emin)
    if not hp:
        s = starts.astype(np.float64); w = np.exp2(es.astype(np.float64))
        f0 = f_np(which, np.maximum(s, lo), F); fh = f_np(which, s + w / 2, F); f1 = f_np(which, s + w, F)
        c2 = 2 * (f1 - 2 * fh + f0); c1 = f1 - f0 - c2
        sc = 2.0 ** g
        C[:, 0] = np.rint(f0 * sc); C[:, 1] = np.rint(c1 * sc); C[:, 2] = np.rint(c2 * sc)
        C[es == 0, 1] = 0; C[es == 0, 2] = 0
        if which == "m":
            C[starts == 0] = 0
        return C
    for i, (s, e) in enumerate(zip(starts, es)):
        s = int(s); e = int(e); w = 1 << e
        if s == 0 and which == "m":
            continue
        f0 = f_mp(which, s, F)
        if e == 0:
            C[i, 0] = int(mnint(f0 * scale)); continue
        fh = f_mp(which, s + w // 2, F)
        f1 = f_mp(which, s + w, F)
        c2 = 2 * (f1 - 2 * fh + f0)
        c1 = f1 - f0 - c2
        C[i] = [int(mnint(f0 * scale)), int(mnint(c1 * scale)), int(mnint(c2 * scale))]
    return C

# ----------------------------------------------------------------------------------------
# valutazione intera bit-exact (stessa aritmetica del codice C++/CUDA)
# ----------------------------------------------------------------------------------------
def eval_int(tab, order, g, m, emin, k):
    """tab: voci (L, ENTRY_W). ritorna (R, Vreal, widths) per un array di k (nessun clamp: k < K_CUT)."""
    k = np.asarray(k, dtype=np.int64)
    idx, e, r = seg_index(k, m, emin)
    if order == 1:
        t0 = tab[idx, 0]; d = tab[idx, 1]
        V = np.left_shift(t0, e) + d * r
        sh = e + g
        R = np.right_shift(V + np.left_shift(np.int64(1), sh - 1), sh)
        Vreal = V.astype(np.float64) / np.exp2(sh.astype(np.float64))
        w = dict(V=int(np.abs(V).max()), prod=int(np.abs(d * r).max()), p1=0, sm=0)
    else:
        c0 = tab[idx, 0]; c1 = tab[idx, 1]; c2 = tab[idx, 2]
        half = np.right_shift(np.left_shift(np.int64(1), e), 1)
        p2 = c2 * r
        t = np.right_shift(p2 + half, e)
        sm_ = np.abs(t + c1).max()
        p1 = (t + c1) * r
        t = np.right_shift(p1 + half, e)
        V = c0 + t
        R = np.right_shift(V + (1 << (g - 1)), g)
        Vreal = V.astype(np.float64) / float(1 << g)
        w = dict(V=int(np.abs(V).max()), prod=int(max(np.abs(p2).max(), np.abs(p1).max())),
                 p1=int(np.abs(p2 + half).max()), sm=int(sm_))
    return R, Vreal, w

def exhaustive(which, tab, F, order, g, m, emin, kend, chunk=1 << 22):
    """verifica su tutti i k in [k0, kend). Ritorna statistiche e checksum."""
    k0 = 0 if which == "p" else 1
    st = dict(maxpre=0.0, maxabs=0.0, sumabs=0.0, n=0, mism=0, maxdiff=0, V=0, prod=0, p1=0, sm=0)
    s1 = 0; s2 = 0; MASK = (1 << 64) - 1
    for a in range(k0, kend, chunk):
        k = np.arange(a, min(a + chunk, kend), dtype=np.int64)
        R, Vreal, w = eval_int(tab, order, g, m, emin, k)
        ideal = f_np(which, k, F)
        pre = np.abs(Vreal - ideal)
        st["maxpre"] = max(st["maxpre"], float(pre.max()))
        a_ = np.abs(R - ideal)
        st["maxabs"] = max(st["maxabs"], float(a_.max())); st["sumabs"] += float(a_.sum())
        rr = np.rint(ideal).astype(np.int64)
        st["mism"] += int(np.count_nonzero(R != rr)); st["n"] += len(k)
        st["maxdiff"] = max(st["maxdiff"], int(np.abs(R - rr).max()))
        st["V"] = max(st["V"], w["V"]); st["prod"] = max(st["prod"], w["prod"])
        st["p1"] = max(st["p1"], w["p1"]); st["sm"] = max(st["sm"], w["sm"])
        sel = k >= 1                                      # i checksum partono da k=1
        Ru = R[sel].astype(np.int64).astype(np.uint64); ku = k[sel].astype(np.uint64)
        s1 = (s1 + int(Ru.sum(dtype=np.uint64))) & MASK
        s2 = (s2 + int((ku * Ru).sum(dtype=np.uint64))) & MASK
    st["meanabs"] = st["sumabs"] / st["n"]
    st["chk1"] = s1; st["chk2"] = s2
    return st

def sampled_maxpre(which, tab, F, order, g, m, emin, kcut):
    """errore pre-arrotondamento su un campione per segmento (per scegliere i parametri)."""
    nseg = num_segments(kcut, m, emin)
    starts, es = seg_start(np.arange(nseg), m, emin)
    w = np.left_shift(np.int64(1), es)
    t = np.linspace(0.0, 1.0, 65)[None, :]
    r = np.minimum((w[:, None] * t).astype(np.int64), w[:, None] - 1)
    k = (starts[:, None] + r).reshape(-1)
    k = np.unique(k[(k < kcut) & (k >= (1 if which == "m" else 0))])
    _, Vreal, _ = eval_int(tab, order, g, m, emin, k)
    return float(np.abs(Vreal - f_np(which, k, F)).max())

def build_entries(which, F, order, g, m, emin, kcut, L=None, hp=True):
    """voci finali (L, ENTRY_W): segmenti reali + voci a zero fino a L (L >= nseg+1)."""
    nseg = num_segments(kcut, m, emin)
    L = nseg + 1 if L is None else L
    assert L >= nseg + 1
    raw = build_table(which, F, order, g, m, emin, kcut, hp=hp)
    E = np.zeros((L, ENTRY_W[order]), dtype=np.int64)
    if order == 1:
        E[:nseg, 0] = raw[:nseg]
        E[:nseg, 1] = raw[1:nseg + 1] - raw[:nseg]
    else:
        E[:nseg, :3] = raw
    return E

def feasible(which, F, order, g, target, kcut, m, emin):
    tab = build_entries(which, F, order, g, m, emin, kcut, hp=False)
    return sampled_maxpre(which, tab, F, order, g, m, emin, kcut) <= target

def best_emin(which, F, order, g, target, kcut, m, share_emin):
    """EMIN massimo (=tabella minima) che rispetta l'obiettivo, a M fissato. None se non esiste."""
    if which == "m" or share_emin:
        return 0 if feasible(which, F, order, g, target, kcut, m, 0) else None
    if not feasible(which, F, order, g, target, kcut, m, 0):
        return None
    lo, hi = 0, F                                   # l'errore cresce con EMIN: ricerca binaria
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(which, F, order, g, target, kcut, m, mid): lo = mid
        else: hi = mid - 1
    return lo

def choose_params(F, order, g, target, kcut, share_emin, mhi=12, mmin=0):
    """M COMUNE a f+ e f- (la valutazione non deve selezionarlo), EMIN per caso. Minimizza i byte totali."""
    best = None
    for m in range(mmin, mhi + 1):
        em = best_emin("m", F, order, g, target, kcut, m, share_emin)
        if em is None: continue
        ep = best_emin("p", F, order, g, target, kcut, m, share_emin)
        if ep is None: continue
        b = table_bytes(order, num_segments(kcut, m, ep)) + table_bytes(order, num_segments(kcut, m, em))
        if best is None or b < best[0]:
            best = (b, m, ep, em)
    if best is None:
        raise RuntimeError("nessun (m, emin) raggiunge l'errore obiettivo %g" % target)
    return best[1], best[2], best[3]

# ----------------------------------------------------------------------------------------
# controllo della funzione ideale float64 contro mpmath
# ----------------------------------------------------------------------------------------
def selfcheck_mp(F, kcut, n=3000, seed=1):
    rng = np.random.default_rng(seed)
    ks = np.unique(np.concatenate([np.arange(1, 64),
                                   np.exp(rng.uniform(0, math.log(kcut - 1), n)).astype(np.int64)]))
    worst = 0.0
    for which in ("p", "m"):
        a = f_np(which, ks, F)
        for kk, v in zip(ks, a):
            worst = max(worst, abs(float(f_mp(which, int(kk), F)) - v))
    return worst

# ----------------------------------------------------------------------------------------
# emissione header
# ----------------------------------------------------------------------------------------
def fmt_entries(name, E, qual, per_line=4):
    w = E.shape[1]
    rows = ["{" + ", ".join("%d" % int(x) for x in row) + "}" for row in E]
    lines = []
    for i in range(0, len(rows), per_line):
        lines.append("    " + ", ".join(rows[i:i + per_line]))
    return "%sconst Entry_%s %s[%d] = {\n%s\n};" % (qual, name[0], name[1], len(rows), ",\n".join(lines))

def emit_header(path, F, order, g, target, kc, m, ep_, em_, Ep, Em, kzp, kzm, sp, sm, cmdline):
    w = ENTRY_W[order]
    comb = np.vstack([Ep, Em])
    assert np.abs(comb).max() < 2 ** 31, "voce fuori da int32"
    payload = hashlib.sha256(comb.astype("<i4").tobytes()).hexdigest()
    r0 = int(eval_int(Ep, order, g, m, ep_, np.array([0]))[0][0])
    out = []
    out.append("// ======================================================================================")
    out.append("// S3_tables_%d_%d.cuh -- GENERATO da gen_s3_tables.py. NON MODIFICARE A MANO." % (32 if F > 16 else 16, F))
    out.append("// Comando: %s" % cmdline)
    out.append("// Parametri: F=%d  ORDER=%d  G=%d  errore obiettivo (pre-arrotondamento) <= %g LSB" % (F, order, g, target))
    out.append("// K_CUT=%d: la valutazione limita k a K_CUT; la voce di indice idx(K_CUT) e' a zero => R+ = R- = 0 per k >= K_CUT" % kc)
    out.append("// M=%d (comune)   voci da %d B (%s)   tabella unica: [f+: %d voci][f-: %d voci] = %d B" %
               (m, 4 * w, "T, dT" if order == 1 else "c0, c1, c2, pad", Ep.shape[0], Em.shape[0], comb.size * 4))
    out.append("// f+ (somma)       : EMIN=%d  errore pre-arr. max %.4f LSB   |R-ideale| medio %.4f max %.4f" % (ep_, sp["maxpre"], sp["meanabs"], sp["maxabs"]))
    out.append("// f- (sottrazione) : EMIN=%d  errore pre-arr. max %.4f LSB   |R-ideale| medio %.4f max %.4f" % (em_, sm["maxpre"], sm["meanabs"], sm["maxabs"]))
    out.append("// Risultati arrotondati diversi dall'ideale: f+ %d/%d (%.4f%%, |diff| max %d)   f- %d/%d (%.4f%%, |diff| max %d)" %
               (sp["mism"], sp["n"], 100.0 * sp["mism"] / sp["n"], sp["maxdiff"],
                sm["mism"], sm["n"], 100.0 * sm["mism"] / sm["n"], sm["maxdiff"]))
    out.append("// Grandezze richieste dalla valutazione (|V| max, |prodotto| max): f+ %d, %d   f- %d, %d" %
               (sp["V"], sp["prod"], sm["V"], sm["prod"]))
    out.append("// sha256 dei dati: %s" % payload)
    out.append("// ======================================================================================")
    out.append("#pragma once")
    out.append("#include \"S3_lut_common.cuh\"")
    out.append("")
    out.append("namespace s3lut {")
    out.append("")
    if order == 1:
        out.append("struct alignas(8) Entry_%d  { int32_t c0, c1; };            // c0 = T[i], c1 = T[i+1]-T[i]" % F)
    else:
        out.append("struct alignas(16) Entry_%d { int32_t c0, c1, c2, pad; };  // coefficienti del segmento" % F)
    out.append("")
    out.append("#ifdef __CUDACC__")
    out.append(fmt_entries((F, "d_tab_%d" % F), comb, "__device__ "))
    out.append("#endif")
    out.append(fmt_entries((F, "h_tab_%d" % F), comb, "static "))
    out.append("")
    out.append("template <> struct Tab<%d> {" % F)
    out.append("    using Entry = Entry_%d;" % F)
    out.append("    static constexpr int      ORDER      = %d;" % order)
    out.append("    static constexpr int      G          = %d;" % g)
    out.append("    static constexpr int      M          = %d;   // comune a f+ e f-" % m)
    out.append("    static constexpr int      EMIN_PLUS  = %d;" % ep_)
    out.append("    static constexpr int      EMIN_MINUS = %d;" % em_)
    out.append("    static constexpr uint32_t K_CUT      = %du;   // k viene limitato a K_CUT (voce a zero)" % kc)
    out.append("    static constexpr uint32_t OFF_MINUS  = %du;   // indice della prima voce di f-" % Ep.shape[0])
    out.append("    static constexpr int32_t  R0_PLUS    = %d;   // R+(0) = round(S*ln 2)" % r0)
    out.append("    // grandezze massime della valutazione intera (per scegliere 32 o 64 bit nel valutatore)")
    out.append("    static constexpr long long VMAX = %d;" % max(sp["V"], sm["V"]))
    out.append("    static constexpr long long PMAX = %d;" % max(sp["prod"], sm["prod"]))
    out.append("    // solo ORDER=2: |c2*r+half| max (1o stadio) e |t+c1| max (fattore del 2o prodotto); 0 se ORDER=1")
    out.append("    static constexpr long long P1MAX = %d;" % max(sp["p1"], sm["p1"]))
    out.append("    static constexpr long long SMAX  = %d;" % max(sp["sm"], sm["sm"]))
    out.append("    // checksum (somma R, somma k*R, mod 2^64) su k = 1 .. K_CUT-1")
    out.append("    static constexpr uint64_t CHK_PLUS_1  = 0x%016xull;" % sp["chk1"])
    out.append("    static constexpr uint64_t CHK_PLUS_2  = 0x%016xull;" % sp["chk2"])
    out.append("    static constexpr uint64_t CHK_MINUS_1 = 0x%016xull;" % sm["chk1"])
    out.append("    static constexpr uint64_t CHK_MINUS_2 = 0x%016xull;" % sm["chk2"])
    out.append("    S3_HD static const Entry* table() {")
    out.append("#ifdef __CUDA_ARCH__")
    out.append("        return d_tab_%d;" % F)
    out.append("#else")
    out.append("        return h_tab_%d;" % F)
    out.append("#endif")
    out.append("    }")
    out.append("};")
    out.append("")
    out.append("} // namespace s3lut")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")

# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--F", type=int); ap.add_argument("--order", type=int, choices=(1, 2))
    ap.add_argument("--g", type=int); ap.add_argument("--target", type=float)
    ap.add_argument("--share-emin", action="store_true", default=None,
                    help="stesso EMIN per f+ e f- (una select in meno nella valutazione, tabella f+ piu' grande)")
    ap.add_argument("--no-share-emin", dest="share_emin", action="store_false")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    cfg = dict(PRESETS[a.preset]) if a.preset else {}
    for key in ("F", "order", "g", "target"):
        if getattr(a, key) is not None:
            cfg[key] = getattr(a, key)
    if set(cfg) - {"share_emin"} != {"F", "order", "g", "target"}:
        ap.error("servono --preset oppure tutti di --F --order --g --target")
    if a.share_emin is not None:
        cfg["share_emin"] = a.share_emin
    F, order, g, target = cfg["F"], cfg["order"], cfg["g"], cfg["target"]
    share = cfg.get("share_emin", False)
    t0 = time.time()
    kcut = compute_kcut(F)
    print("F=%d order=%d g=%d target=%g LSB share_emin=%s   K_CUT(reale)=%d (2^%.2f)" % (F, order, g, target, share, kcut, math.log2(kcut)))
    print("controllo f64 contro mpmath (campione) : max |diff| = %.2e LSB" % selfcheck_mp(F, kcut))

    m, ep_, em_ = choose_params(F, order, g, target, kcut, share)
    while True:
        nsp = num_segments(kcut, m, ep_); nsm = num_segments(kcut, m, em_)
        kzp = int(seg_start(np.array([nsp]), m, ep_)[0][0]); kzm = int(seg_start(np.array([nsm]), m, em_)[0][0])
        kc = max(kzp, kzm)                                   # limite comune di k
        Lp = int(seg_index(np.array([kc]), m, ep_)[0][0]) + 1
        Lm = int(seg_index(np.array([kc]), m, em_)[0][0]) + 1
        Ep = build_entries("p", F, order, g, m, ep_, kcut, L=Lp, hp=True)
        Em = build_entries("m", F, order, g, m, em_, kcut, L=Lm, hp=True)
        sp = exhaustive("p", Ep, F, order, g, m, ep_, kzp)
        sm = exhaustive("m", Em, F, order, g, m, em_, kzm)
        if max(sp["maxpre"], sm["maxpre"]) <= target:
            break
        m += 1                                               # il campione l'aveva mancato: infittisci
        ep_ = best_emin("p", F, order, g, target, kcut, m, share); em_ = best_emin("m", F, order, g, target, kcut, m, share)
        assert ep_ is not None and em_ is not None
    # k in [kzX, kc): voci a zero; il valore reale e' < 0,5 (k >= K_CUT reale) => R = 0 corretto
    for nm, st, tab, kz, mm_ in (("f+", sp, Ep, kzp, ep_), ("f-", sm, Em, kzm, em_)):
        print("%s: M=%d EMIN=%d voci=%d (%d B) | pre-arr. max %.4f | |R-ideale| medio %.4f max %.4f | "
              "diversi dall'ideale arrotondato %d/%d (%.4f%%) max|diff|=%d | |V| max %d, prodotto max %d"
              % (nm, m, mm_, tab.shape[0], tab.size * 4, st["maxpre"], st["meanabs"], st["maxabs"], st["mism"], st["n"],
                 100.0 * st["mism"] / st["n"], st["maxdiff"], st["V"], st["prod"]))
    print("K_CUT (limite di k) = %d  (reale %d)   OFF_MINUS=%d" % (kc, kcut, Lp))
    name = "S3_tables_%d_%d.cuh" % (32 if F > 16 else 16, F)
    path = os.path.join(a.outdir, name)
    emit_header(path, F, order, g, target, kc, m, ep_, em_, Ep, Em, kzp, kzm, sp, sm,
                " ".join(["python3 gen_s3_tables.py"] + sys.argv[1:]))
    print("scritto %s   (totale %d B, %.1f s)" % (path, (Ep.size + Em.size) * 4, time.time() - t0))

if __name__ == "__main__":
    main()

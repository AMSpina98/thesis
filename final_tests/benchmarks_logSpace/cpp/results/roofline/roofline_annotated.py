#!/usr/bin/env python3
"""Una figura annotata per benchmark: roofline FLOP + roofline di istruzioni + tabella.

Uso:  python roofline_annotated.py metrics.csv <cartella_output>
Produce roofline_annotata_<benchmark>.png (vectorAdd, matrixMul, customVecMatMul, reduction, transpose).

Colonna "Limite probabile" (euristica, nell'ordine):
  FP64 pipe >= 70%            -> unita' FP64 sature
  throughput L1 >= 80%        -> L1 / shared memory
  banda DRAM >= 70% di 192    -> banda DRAM
  issue >= 55% del picco      -> emissione istruzioni
  altrimenti                  -> latenza / altro
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

src, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)

SMS, ISSUE_IPC = 16, 4.0
FP32_PEAK, FP64_PEAK, BW = 3041.0, 95.0, 192.0
FLOPS = {"vectorAdd": 50000.0, "matrixMul": 2.0 * 320 * 320 * 640,
         "customVecMatMul": 2.0 * 5000 * 5000, "reduction": float(2 ** 24 - 1)}
TITLES = {"vectorAdd": "Vector Add (N = 50 000)", "matrixMul": "Matrix Mul (320×320 · 640×320)",
          "customVecMatMul": "Vector-Matrix Mul (5000×5000)", "reduction": "Reduction (N = 2²⁴)",
          "transpose": "Transpose (naive e ottimizzato)"}

INK, SOFT, GRID = "#0b0b0b", "#52514e", "#d9d8d2"
C_BASE, C_ENC, C_S2, C_S3 = "#52514e", "#1baf7a", "#2a78d6", "#eb6834"
TYPES = [("float", "float", C_BASE, "o"), ("double", "double", C_BASE, "s"),
         ("encfp", "Enc<float>", C_ENC, "D"), ("encdb", "Enc<double>", C_ENC, "X"),
         ("s2_16_10", "S2 16_10", C_S2, "o"), ("s2_16_11", "S2 16_11", C_S2, "s"),
         ("s2_32_23", "S2 32_23", C_S2, "^"), ("s3_16_10", "S3 16_10", C_S3, "o"),
         ("s3_32_21", "S3 32_21", C_S3, "^")]
TINFO = {k: (lab, col, mk) for k, lab, col, mk in TYPES}


def fl(r, k):
    v = r.get(k)
    return float(v) if v not in (None, "", "None") else None


def variant_of(name):
    for k, *_ in sorted(TYPES, key=lambda t: -len(t[0])):
        if name.endswith("_" + k):
            return k
    return None


def load():
    data = {}
    with open(src, newline="") as f:
        for r in csv.DictReader(f):
            name, bench = r["file"], r["bench"]
            ver = None
            if bench == "transpose":
                for suf in ("naive", "opt"):
                    if name.endswith("_" + suf):
                        name, ver = name[: -len(suf) - 1], suf
            v = variant_of(name)
            if v is None:
                continue
            t = fl(r, "time")
            f_ghz = fl(r, "f")
            winst = fl(r, "winst")
            dram = fl(r, "dr") + fl(r, "dw")
            ipc = winst / (t * 1e-3 * f_ghz * 1e9) / SMS
            m = dict(v=v, t=t, winst=winst, dram=dram, pct_issue=100 * ipc / ISSUE_IPC,
                     l1=fl(r, "l1thr"), fp64=fl(r, "fp64") or 0.0,
                     gips=winst / (t * 1e-3) / 1e9, bw=dram / (t * 1e-3) / 1e9)
            if bench in FLOPS:
                m["perf"] = FLOPS[bench] / (t * 1e-3) / 1e9
                m["ai"] = FLOPS[bench] / dram
            m["ai_i"] = winst / (dram / 32.0)
            data.setdefault(bench if ver is None else (bench, ver), {})[v] = m
    return data


NOTE = ("* Euristica: FP64 se pipe FP64 ≥ 70%; L1 se throughput L1 ≥ 80%; banda DRAM se ≥ 70% di 192 GB/s; "
        "emissione se ≥ 55% del picco di issue; altrimenti latenza/altro. Picco issue = 4 warp-inst/ciclo/SM (teorico).")


SH = {"banda DRAM": "banda DRAM", "L1 / shared": "L1", "latenza / altro": "latenza", "emissione istr.": "emissione", "FP64": "FP64"}


def limit(m):
    if m["fp64"] >= 70: return "FP64"
    if m["l1"] >= 80: return "L1 / shared"
    if m["bw"] >= 0.7 * BW: return "banda DRAM"
    if m["pct_issue"] >= 55: return "emissione istr."
    return "latenza / altro"


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors=SOFT, labelsize=8.5)


def pts(ax, d, xk, yk):
    for k, m in d.items():
        lab, col, mk = TINFO[k]
        ax.scatter([m[xk]], [m[yk]], s=85, color=col, marker=mk, edgecolor="white", linewidth=0.9, zorder=5)


def panel_flop(ax, d, title):
    style(ax)
    xs = np.logspace(-2, 4, 200)
    ax.plot(xs, np.minimum(xs * BW, FP32_PEAK), color=INK, lw=1.5, zorder=3)
    ax.hlines(FP32_PEAK, FP32_PEAK / BW, 1e4, color=INK, lw=1.5, zorder=3)
    ax.hlines(FP64_PEAK, FP64_PEAK / BW, 1e4, color=INK, lw=1.1, ls="--", zorder=3)
    ax.text(0.9e4, FP32_PEAK * 1.08, "FP32 3041 GFLOP/s", ha="right", fontsize=8, color=INK)
    ax.text(0.9e4, FP64_PEAK * 1.12, "FP64 95 GFLOP/s", ha="right", fontsize=8, color=INK)
    ax.text(0.0125, 1.7, "diagonale = banda 192 GB/s\n(a sinistra del ginocchio il limite è la memoria)",
            fontsize=7.5, color=SOFT, style="italic", va="top")
    pts(ax, d, "ai", "perf")
    # freccia: overhead di S3_32_21 rispetto a float (altezza = velocita')
    if "float" in d and "s3_32_21" in d:
        a, b = d["float"], d["s3_32_21"]
        ax.annotate("", xy=(b["ai"], b["perf"] * 1.12), xytext=(a["ai"], a["perf"] * 0.88),
                    arrowprops=dict(arrowstyle="->", color=C_S3, lw=1.3, ls="--",
                                    connectionstyle="arc3,rad=0.15"), zorder=4)
        ax.text(a["ai"] * 0.55, a["perf"] * 2.6,
                f"S3 32_21: ×{b['t'] / a['t']:.1f} più lento di float", fontsize=8.5, color=C_S3,
                fontweight="bold", ha="left")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-2, 1e4); ax.set_ylim(1e-1, 6e3)
    ax.set_xlabel("FLOP nominali / byte DRAM", fontsize=9, color=SOFT)
    ax.set_ylabel("GFLOP/s nominali (= velocità)", fontsize=9, color=SOFT)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, loc="left")


def panel_instr(ax, d, title, with_ylabel=True):
    style(ax)
    peak = SMS * ISSUE_IPC * 1.483
    bw_s = BW / 32.0
    ridge = peak / bw_s
    xs = np.logspace(-1, 6, 300)
    ax.axvspan(1e-1, ridge, color="#ecebe6", alpha=0.6, zorder=0)
    ax.plot(xs, np.minimum(xs * bw_s, peak), color=INK, lw=1.5, zorder=3)
    ax.hlines(peak, ridge, 1e7, color=INK, lw=1.5, zorder=3)
    ax.text(ridge * 1.25, peak * 1.09, f"emissione istruzioni: {peak:.0f} G warp-inst/s", ha="left", fontsize=8, color=INK)
    ax.text(0.12, 0.3, f"banda DRAM {BW:.0f} GB/s", fontsize=8, color=SOFT)
    ax.text(0.12, 230, "zona limitata dalla memoria", fontsize=7.5, color=SOFT, style="italic", va="top")
    ax.text(3e5, 230, "zona limitata dall'emissione\n(o dalla L1 / latenza)", fontsize=7.5, color=SOFT,
            style="italic", va="top", ha="right")
    pts(ax, d, "ai_i", "gips")
    for k in ("s3_32_21",):
        if k in d:
            m = d[k]
            lab, col, _ = TINFO[k]
            txt = f"{lab}\n{m['pct_issue']:.0f}% issue · L1 {m['l1']:.0f}%"
            off = (10, -26) if k == "s3_32_21" else (10, 8)
            ax.annotate(txt, (m["ai_i"], m["gips"]), xytext=off, textcoords="offset points", fontsize=7.5,
                        color=C_S3 if k.startswith("s3") else SOFT, fontweight="bold", zorder=6,
                        arrowprops=dict(arrowstyle="-", color="#8a8984", lw=0.7))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-1, 1e6); ax.set_ylim(1e-1, 3e2)
    ax.set_xlabel("warp-istruzioni / settore DRAM (32 B)", fontsize=9, color=SOFT)
    if with_ylabel: ax.set_ylabel("G warp-istruzioni / s", fontsize=9, color=SOFT)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, loc="left")


def legend(fig, present):
    hs = [Line2D([], [], marker=mk, ls="", color=col, markeredgecolor="white", markeredgewidth=0.8,
                 markersize=9, label=lab) for k, lab, col, mk in TYPES if k in present]
    fig.legend(handles=hs, loc="upper center", ncol=len(hs), fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.955), handletextpad=0.2, columnspacing=1.3)


def table(ax, d, cols, rowfn):
    ax.axis("off")
    rows = [rowfn(k, d[k]) for k, *_ in TYPES if k in d]
    tb = ax.table(cellText=[r[0] for r in rows], colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.35)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(GRID)
        if i == 0:
            c.set_facecolor("#ecebe6"); c.set_text_props(fontweight="bold", color=INK)
        else:
            c.set_facecolor("#fcfcfb")
            if j == 0:
                c.set_text_props(color=rows[i - 1][1], fontweight="bold")
    return tb


data = load()
for bench in ("vectorAdd", "matrixMul", "customVecMatMul", "reduction"):
    d = data[bench]
    tf = d["float"]["t"]
    fig = plt.figure(figsize=(13, 9.4), dpi=170)
    fig.patch.set_facecolor("#fcfcfb")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 0.85], hspace=0.16, wspace=0.18,
                          left=0.07, right=0.97, top=0.88, bottom=0.03)
    fig.suptitle(f"Roofline — {TITLES[bench]}", fontsize=15, fontweight="bold", color=INK, x=0.07, ha="left", y=0.985)
    legend(fig, set(d))
    panel_flop(fig.add_subplot(gs[0, 0]), d, "Roofline FLOP (altezza = velocità del tipo)")
    panel_instr(fig.add_subplot(gs[0, 1]), d, "Roofline di istruzioni (quanto emette la GPU)", with_ylabel=False)

    def row(k, m):
        lab, col, _ = TINFO[k]
        print("LIM", bench, lab, limit(m), f"issue={m['pct_issue']:.0f} L1={m['l1']:.0f} bw={m['bw']:.0f}")
        return ([lab, f"{m['t']:.4g}", f"×{m['t'] / tf:.2f}", (f"{m['winst'] / 1e6:.1f}" if m['winst'] >= 1e7 else f"{m['winst'] / 1e6:.2f}"), f"{m['pct_issue']:.0f}%",
                 f"{m['l1']:.0f}%", f"{m['bw']:.0f}", limit(m)], col)

    table(fig.add_subplot(gs[1, :]), d,
          ["Tipo", "Tempo (ms)", "vs float", "Warp-inst (M)", "% picco issue", "Throughput L1", "DRAM (GB/s)",
           "Limite probabile*"], row)
    fig.text(0.07, 0.012, NOTE, fontsize=7.5, color=SOFT, ha="left", va="bottom")
    fig.savefig(os.path.join(outdir, f"roofline_annotata_{bench}.png"), facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------------------------------------------------ transpose: naive | ottimizzato
dn, do = data[("transpose", "naive")], data[("transpose", "opt")]
fig = plt.figure(figsize=(13, 9.4), dpi=170)
fig.patch.set_facecolor("#fcfcfb")
gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 0.85], hspace=0.16, wspace=0.18,
                      left=0.07, right=0.97, top=0.88, bottom=0.03)
fig.suptitle(f"Roofline di istruzioni — {TITLES['transpose']}", fontsize=15, fontweight="bold", color=INK,
             x=0.07, ha="left", y=0.985)
legend(fig, set(do))
panel_instr(fig.add_subplot(gs[0, 0]), dn, "Naive (accessi non coalesciati)")
panel_instr(fig.add_subplot(gs[0, 1]), do, "Ottimizzato (shared memory, senza conflitti)", with_ylabel=False)


def row_t(k, m):
    lab, col, _ = TINFO[k]
    n = dn[k]
    return ([lab, f"{n['t']:.4g}", f"{n['bw']:.0f}", f"{m['t']:.4g}", f"{m['bw']:.0f}", f"{100 * m['bw'] / BW:.0f}%",
             f"×{n['t'] / m['t']:.2f}", SH.get(limit(n), limit(n)) + " → " + SH.get(limit(m), limit(m))], col)


table(fig.add_subplot(gs[1, :]), do,
      ["Tipo", "Naive (ms)", "Naive GB/s", "Ottimiz. (ms)", "Ottimiz. GB/s", "% di 192 GB/s", "Speedup", "Limite: naive → ottimiz.*"],
      row_t)
fig.text(0.07, 0.012, NOTE, fontsize=7.5, color=SOFT, ha="left", va="bottom")
fig.savefig(os.path.join(outdir, "roofline_annotata_transpose.png"), facecolor=fig.get_facecolor())
plt.close(fig)
print("ok:", sorted(os.listdir(outdir)))

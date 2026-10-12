#!/usr/bin/env python3
"""Roofline dai CSV ncu (raw) gia' estratti in metrics.csv.

Uso:  python roofline.py metrics.csv <cartella_output>

Produce:
  roofline_flop_<bench>.png   roofline classica (FLOP algoritmici / byte DRAM), come slide 44-47
  roofline_instr_<bench>.png  roofline di istruzioni (warp-inst / settore DRAM), adatta ai tipi interi
  roofline_summary.csv        tabella riassuntiva per la discussione
Tipi non validi (S3 in reduction, S3 in transpose finche' non rilanciati) vengono saltati.
"""
import csv, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

src, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)

# ---------------------------------------------------------------- hardware (GTX 1650 Ti, ncu, clock bloccati)
SMS = 16
FP32_PEAK = 3041.0          # GFLOP/s   (come nel PDF: 1024 core x 2 x 1.485 GHz)
FP64_PEAK = 95.0            # GFLOP/s
BW_NOM = 192.0              # GB/s nominale
ISSUE_IPC = 4.0             # warp-inst / ciclo / SM (4 scheduler)

# ---------------------------------------------------------------- problemi (FLOP algoritmici nominali)
FLOPS = {
    "vectorAdd":      50000.0,                 # N somme, N = 50000
    "matrixMul":      2.0 * 320 * 320 * 640,   # A 320x320 * B 640x320
    "customVecMatMul": 2.0 * 5000 * 5000,      # matrice 5000x5000 * vettore
    "reduction":      float(2 ** 24 - 1),      # n-1 somme, n = 2^24
}
TITLES = {"vectorAdd": "Vector Add", "matrixMul": "Matrix Mul",
          "customVecMatMul": "Vector-Matrix Mul", "reduction": "Reduction",
          "transpose_naive": "Transpose (naive)", "transpose_opt": "Transpose (ottimizzato)"}

# ---------------------------------------------------------------- stile: colore = famiglia, forma = tipo
INK, GRID = "#0b0b0b", "#d9d8d2"
C_BASE, C_ENC, C_S2, C_S3 = "#52514e", "#1baf7a", "#2a78d6", "#eb6834"
TYPES = [  # chiave nel nome file, etichetta, colore, marker
    ("float",    "float",        C_BASE, "o"),
    ("double",   "double",       C_BASE, "s"),
    ("encfp",    "Enc<float>",   C_ENC,  "D"),
    ("encdb",    "Enc<double>",  C_ENC,  "X"),
    ("s2_16_10", "S2 16_10",     C_S2,   "o"),
    ("s2_16_11", "S2 16_11",     C_S2,   "s"),
    ("s2_32_23", "S2 32_23",     C_S2,   "^"),
    ("s3_16_10", "S3 16_10",     C_S3,   "o"),
    ("s3_32_21", "S3 32_21",     C_S3,   "^"),
]
LAB = {k: l for k, l, _, _ in TYPES}


def load():
    rows = []
    with open(src, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def f(r, k):
    v = r.get(k)
    return float(v) if v not in (None, "", "None") else None


def variant_of(file):
    for k, *_ in sorted(TYPES, key=lambda t: -len(t[0])):
        if file.endswith("_" + k) or ("_" + k + "_") in file:
            return k
    return None


def valid(r, v):
    # reduction S3: il CSV e' ancora reduce2<int>; transpose S3 assente
    return not (r["bench"] == "reduction" and v.startswith("s3") and "int>" in r["kernel"])


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=9)


def legend_handles(present):
    hs = []
    for k, lab, col, mk in TYPES:
        if k in present:
            hs.append(Line2D([], [], marker=mk, ls="", color=col, markeredgecolor="white",
                             markeredgewidth=0.8, markersize=9, label=lab))
    return hs


def scatter(ax, pts):
    for k, x, y in pts:
        _, lab, col, mk = next(t for t in TYPES if t[0] == k)
        ax.scatter([x], [y], s=95, color=col, marker=mk, edgecolor="white", linewidth=0.9, zorder=5)


def label_some(ax, pts, keys=("s3_32_21", "float")):
    off = {"s3_32_21": (9, -15), "float": (8, 7)}
    for k, x, y in pts:
        if k in keys:
            ax.annotate(LAB[k], (x, y), xytext=off.get(k, (7, 6)), textcoords="offset points",
                        fontsize=8.5, color=C_S3 if k.startswith("s3") else "#52514e",
                        fontweight="bold", zorder=6)


rows = load()
by_bench = {}
for r in rows:
    name = r["file"]
    bench = r["bench"]
    for suf in ("_naive", "_opt"):
        if bench == "transpose" and name.endswith(suf):
            name = name[: -len(suf)]
            bench = "transpose" + suf
    r = dict(r, file=name, bench=bench)
    v = variant_of(name)
    if v is None or not valid(r, v):
        continue
    by_bench.setdefault(bench, {})[v] = r

summary = []
for bench, d in by_bench.items():
    has_flop = bench in FLOPS
    t_float = f(d["float"], "time") if "float" in d else None
    ptsF, ptsI = [], []
    for v, r in d.items():
        t = f(r, "time") * 1e-3                     # s
        dram = f(r, "dr") + f(r, "dw")              # byte
        winst = f(r, "winst")
        flop = FLOPS.get(bench, 0.0)
        perf = flop / t / 1e9                       # GFLOP/s (nominali)
        ai = flop / dram
        gips = winst / t / 1e9                      # G warp-inst/s
        ai_i = winst / (dram / 32.0)                # warp-inst / settore DRAM
        ptsF.append((v, ai, perf))
        ptsI.append((v, ai_i, gips))
        fq = f(r, "f")
        ipc = winst / (f(r, "time") * 1e-3 * fq * 1e9) / SMS
        summary.append(dict(bench=bench, tipo=LAB[v], tempo_ms=round(f(r, "time"), 4),
                            vs_float=round(f(r, "time") / t_float, 2) if t_float else "",
                            GFLOPs_nominali=round(perf, 2), AI_FLOP_per_byte=round(ai, 2),
                            warp_inst_M=round(winst / 1e6, 2), GIPS=round(gips, 2),
                            IPC_per_SM=round(ipc, 3), pct_issue_peak=round(100 * ipc / ISSUE_IPC, 1),
                            L1_thr_pct=round(f(r, "l1thr"), 1), L1_sector_M=round(f(r, "l1sec") / 1e6, 2),
                            DRAM_MB=round(dram / 1e6, 3), ALU_pct=round(f(r, "alu"), 1),
                            FMA_pct=round(f(r, "fma"), 1), LSU_pct=round(f(r, "lsu"), 1)))
    present = set(d)

    # ----------------------------------------------------------- 1) roofline classica FLOP
    if has_flop:
        fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=200)
        fig.patch.set_facecolor("#fcfcfb")
        style(ax)
        xs = np.logspace(-2, 4, 200)
        ax.plot(xs, np.minimum(xs * BW_NOM, FP32_PEAK), color=INK, lw=1.6, zorder=3)
        ax.hlines(FP32_PEAK, FP32_PEAK / BW_NOM, 1e4, color=INK, lw=1.6, zorder=3)
        ax.hlines(FP64_PEAK, FP64_PEAK / BW_NOM, 1e4, color=INK, lw=1.2, ls="--", zorder=3)
        ax.text(1e4 * 0.9, FP32_PEAK * 1.08, f"FP32  {FP32_PEAK:.0f} GFLOP/s", ha="right", fontsize=8.5, color=INK)
        ax.text(1e4 * 0.9, FP64_PEAK * 1.12, f"FP64  {FP64_PEAK:.0f} GFLOP/s", ha="right", fontsize=8.5, color=INK)
        ax.text(0.0125, 1.7, "banda 192 GB/s", fontsize=8.5, color="#52514e", va="top")
        scatter(ax, ptsF)
        label_some(ax, ptsF)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1e-2, 1e4); ax.set_ylim(1e-1, 1e4)
        ax.set_xlabel("Intensità aritmetica (FLOP nominali / byte DRAM)", fontsize=10, color="#52514e")
        ax.set_ylabel("Prestazione (GFLOP/s nominali)", fontsize=10, color="#52514e")
        ax.set_title(f"Roofline — {TITLES[bench]}", fontsize=13, fontweight="bold", color=INK, loc="left")
        ax.legend(handles=legend_handles(present), loc="lower right", fontsize=8.5, frameon=True,
                  framealpha=0.95, edgecolor=GRID, ncol=1, title="Tipo", title_fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"roofline_flop_{bench}.png"), facecolor=fig.get_facecolor())
        plt.close(fig)

    # ----------------------------------------------------------- 2) roofline di istruzioni
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    style(ax)
    peak_gips = SMS * ISSUE_IPC * 1.483      # G warp-inst/s
    bw_sect = BW_NOM / 32.0                  # Gsector/s
    xs = np.logspace(-1, 6, 300)
    ax.plot(xs, np.minimum(xs * bw_sect, peak_gips), color=INK, lw=1.6, zorder=3)
    ax.hlines(peak_gips, peak_gips / bw_sect, 1e7, color=INK, lw=1.6, zorder=3)
    ax.text(1e6, peak_gips * 1.08, f"issue  {peak_gips:.0f} G warp-inst/s", ha="right", fontsize=8.5, color=INK)
    ax.text(0.15, 0.15 * bw_sect * 1.6, f"DRAM {BW_NOM:.0f} GB/s", fontsize=8, color="#52514e")
    scatter(ax, ptsI)
    label_some(ax, ptsI)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-1, 1e6); ax.set_ylim(1e-1, 3e2)
    ax.set_xlabel("Warp-istruzioni per settore DRAM (32 B)", fontsize=10, color="#52514e")
    ax.set_ylabel("G warp-istruzioni / s", fontsize=10, color="#52514e")
    ax.set_title(f"Roofline di istruzioni — {TITLES[bench]}", fontsize=13, fontweight="bold", color=INK, loc="left")
    ax.legend(handles=legend_handles(present), loc="lower right", fontsize=8.5, frameon=True,
              framealpha=0.95, edgecolor=GRID, title="Tipo", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"roofline_instr_{bench}.png"), facecolor=fig.get_facecolor())
    plt.close(fig)

with open(os.path.join(outdir, "roofline_summary.csv"), "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(summary[0].keys()))
    w.writeheader(); w.writerows(summary)
print("ok", len(summary), "righe;", sorted(by_bench))

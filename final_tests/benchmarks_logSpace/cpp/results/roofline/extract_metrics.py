#!/usr/bin/env python3
"""Estrae da results/<bench>/csv/*.csv (ncu raw) le metriche per la roofline.
Uso (dalla cartella results):  python roofline/extract_metrics.py
Scrive roofline/metrics.csv: una riga per file, media sulle invocazioni.
Per ogni file tiene solo le invocazioni con la griglia piu' grande
(in reduction scarta i lanci piccoli del secondo livello)."""
import csv, re, glob, os

def num(s):
    s = s.strip()
    if s == '': return None
    if re.fullmatch(r'-?\d{1,3}(\.\d{3})+', s): return float(s.replace('.', ''))   # separatore migliaia
    try: return float(s.replace(',', '.'))
    except ValueError: return None

TU = {'s': 1e3, 'ms': 1, 'msecond': 1, 'us': 1e-3, 'usecond': 1e-3, 'ns': 1e-6, 'nsecond': 1e-6}   # -> ms
BU = {'byte': 1, 'Kbyte': 1e3, 'Mbyte': 1e6, 'Gbyte': 1e9}                                           # -> byte
M = {'time': 'gpu__time_duration.sum', 'f': 'sm__cycles_elapsed.avg.per_second', 'winst': 'smsp__inst_executed.sum',
     'l1sec': 'l1tex__t_sectors.sum', 'l2sec': 'lts__t_sectors.sum', 'dr': 'dram__bytes_read.sum', 'dw': 'dram__bytes_write.sum',
     'l1thr': 'l1tex__throughput.avg.pct_of_peak_sustained_elapsed', 'smthr': 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
     'dramthr': 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
     'alu': 'sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed', 'fma': 'sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed',
     'fp64': 'sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_elapsed', 'lsu': 'sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed',
     'xu': 'sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed', 'thr': 'launch__thread_count', 'l1hit': 'l1tex__t_sector_hit_rate.pct'}
GRID = 'launch__grid_size'
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
keys = ['bench', 'file', 'n', 'n_tot', 'kernel'] + list(M)
out = []
for f in sorted(glob.glob('*/csv/*.csv')):
    bench, name = f.split('/')[0], os.path.basename(f)[:-4]
    r = list(csv.reader(open(f, encoding='utf-8-sig')))
    h, u, d = r[0], r[1], r[2:]
    if GRID in h:
        g = [num(x[h.index(GRID)]) or 0 for x in d]
        gm = max(g); d = [x for x, gg in zip(d, g) if gg == gm]
    rec = {'bench': bench, 'file': name, 'n': len(d), 'n_tot': len(r) - 2, 'kernel': d[0][4]}
    for k, col in M.items():
        if col not in h: continue
        i = h.index(col); vals = []
        for row in d:
            x = num(row[i])
            if x is None: continue
            un = u[i].strip()
            if k == 'time': x *= TU.get(un, 1)
            if k in ('dr', 'dw'): x *= BU.get(un, 1)
            if k == 'f': x *= {'Ghz': 1, 'GHz': 1, 'Mhz': 1e-3}.get(un, 1)
            vals.append(x)
        rec[k] = sum(vals) / len(vals) if vals else None
    out.append(rec)
with open('roofline/metrics.csv', 'w', newline='') as fo:
    w = csv.writer(fo); w.writerow(keys)
    for rec in out: w.writerow([rec.get(k) for k in keys])
print(len(out), 'file ->', 'roofline/metrics.csv')
for rec in out:
    if rec['n'] != rec['n_tot'] or rec['n'] < 5: print('  ', rec['file'], 'righe usate', rec['n'], 'di', rec['n_tot'])

#!/usr/bin/env python3
"""Regenerates Figs. 1-5 of the revised manuscript from the raw records.
Usage: python3 figures.py <data_dir> <offline_results.json> <out_dir> [revision_T4.json revision_P100.json]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA, OFF, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
REV = [json.load(open(p)) for p in sys.argv[4:] if os.path.exists(p)]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                     "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                     "mathtext.fontset": "stix", "savefig.dpi": 300})
L = lambda n: json.load(open(os.path.join(DATA, n)))
T4, P100 = L("final_Tesla_T4.json"), L("final_Tesla_P100_PCIE_16GB.json")
TPU, TPUA = L("gate_tpu_TPU_v5_lite.json"), L("tpu_addendum.json")
ALG = {"T4": L("algos_Tesla_T4.json"), "P100": L("algos_Tesla_P100_PCIE_16GB.json")}
off = json.load(open(OFF))
C = {"T4": "#1f4e79", "P100": "#c0504d", "TPU": "#4f8a3a", "host": "#7f7f7f"}
M = {"T4": "o", "P100": "s", "TPU": "^"}


def save(name):
    plt.tight_layout(pad=0.4)
    plt.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------- Fig. 1
plt.figure(figsize=(4.6, 2.3))
for lab, curve, col, mk in (("Host CPU (Xeon 2.00 GHz)", T4["bandwidth"]["cpu"]["curve"], C["host"], "D"),
                            ("Tesla T4", T4["bandwidth"]["gpu"]["curve"], C["T4"], "o"),
                            ("Tesla P100", P100["bandwidth"]["gpu"]["curve"], C["P100"], "s"),
                            ("TPU v5e", TPUA["bandwidth"]["curve"], C["TPU"], "^")):
    x = [r["working_set_bytes"] / 1e6 for r in curve]
    y = [r["bw_bytes_per_s"] / 1e9 for r in curve]
    plt.plot(x, y, marker=mk, ms=3, color=col, label=lab)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Working-set size (MB)"); plt.ylabel("Achieved bandwidth (GB/s)")
plt.grid(alpha=0.3, which="both", lw=0.4); plt.legend(frameon=False, ncol=2)
save("fig1_bandwidth")

# ---------------------------------------------------------------- Fig. 2
plt.figure(figsize=(4.6, 2.35))
for dev, d in (("T4", T4), ("P100", P100)):
    for side, face in (("cuda", C[dev]), ("cpu", "none")):
        g = d["grid"][side]
        n = [r["NP"] * r["D"] for r in g]
        t = [r["time_s"] * 1e6 for r in g]
        plt.scatter(n, t, s=9, marker=M[dev], facecolors=face, edgecolors=C[dev], linewidths=0.6,
                    label=f"{dev} {'GPU' if side == 'cuda' else 'host CPU'}")
plt.xscale("log"); plt.yscale("log")
plt.xlabel(r"Work $N = NP \times D$ (elements)"); plt.ylabel(r"Per-generation time ($\mu$s)")
plt.grid(alpha=0.3, which="both", lw=0.4); plt.legend(frameon=False, ncol=2)
save("fig2_per_generation")

# ---------------------------------------------------------------- Fig. 3
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.35))
for dev, rows in (("T4", T4["breakeven_eager"]), ("P100", P100["breakeven_eager"]), ("TPU", TPU["breakeven"])):
    pts = [(r["D"], r["np_star"]) for r in rows if r.get("np_star") and not r.get("censored")]
    D = np.array([p[0] for p in pts], float); inv = 1e3 / np.array([p[1] for p in pts], float)
    co = np.polyfit(D, inv, 1); r2 = off["signature"]["DE_main"][dev]["r2"]
    xs = np.linspace(0, D.max() * 1.03, 50)
    lab = {"T4": "Tesla T4", "P100": "Tesla P100", "TPU": "TPU v5e"}[dev]
    ax[0].plot(D, inv, M[dev], color=C[dev], ms=4, label=f"{lab} ($R^2$ = {r2:.4f})")
    ax[0].plot(xs, np.polyval(co, xs), "-", color=C[dev], lw=0.8)
ax[0].set_xlabel(r"Dimension $D$"); ax[0].set_ylabel(r"$10^3 / NP^*$ (individuals$^{-1}$)")
ax[0].set_title("(a) DE on three devices", fontsize=8)
ax[0].grid(alpha=0.3, lw=0.4); ax[0].legend(frameon=False)
# panel (b): algorithms -- six-dimension revision sweep when available, else the 3-dimension sweep
styles = {"de": "o", "jde": "s", "pso": "^", "cmaes": "v"}
names = {"de": "DE", "jde": "jDE", "pso": "PSO", "cmaes": "CMA-ES"}
used_rev = False
if REV and all("s6_signature" in r for r in REV):
    used_rev = True
    for rv in REV:
        dev = "T4" if "T4" in rv["environment"].get("gpu_name", "") else "P100"
        for al, v in rv["s6_signature"].items():
            pts = [(r["D"], r["np_star"]) for r in v["breakeven"] if r.get("np_star") and not r.get("censored")]
            if len(pts) < 3:
                continue
            D = np.array([p[0] for p in pts], float); inv = 1e3 / np.array([p[1] for p in pts], float)
            co = np.polyfit(D, inv, 1)
            col = {"de": "#1f4e79", "jde": "#c0504d", "pso": "#4f8a3a"}[al]
            ax[1].plot(D, inv, styles[al], color=col, ms=4, label=f"{names[al]} ($R^2$ = {v['signature_r2']:.4f})")
            xs = np.linspace(0, D.max() * 1.03, 50)
            ax[1].plot(xs, np.polyval(co, xs), "-", color=col, lw=0.8)
else:
    for dev in ("T4", "P100"):
        for al, v in ALG[dev]["algorithms"].items():
            pts = [(r["D"], r["np_star"]) for r in v["breakeven"] if r.get("np_star") and not r.get("censored")]
            D = np.array([p[0] for p in pts], float); inv = 1e3 / np.array([p[1] for p in pts], float)
            co = np.polyfit(D, inv, 1)
            ax[1].plot(D, inv, styles[al], color=C[dev], ms=3.5, mfc="none" if dev == "P100" else C[dev],
                       label=f"{names[al]}, {dev}")
            ax[1].plot(D, np.polyval(co, D), "-", color=C[dev], lw=0.6)
ax[1].set_xlabel(r"Dimension $D$"); ax[1].set_ylabel(r"$10^3 / NP^*$ (individuals$^{-1}$)")
ax[1].set_title("(b) DE, jDE and PSO on the T4 (revision run)" if used_rev else "(b) Other methods (T4 filled, P100 open)", fontsize=8)
ax[1].grid(alpha=0.3, lw=0.4); ax[1].legend(frameon=False, ncol=1 if used_rev else 2, fontsize=6.5 if used_rev else 5.5)
save("fig3_signature")

# ---------------------------------------------------------------- Fig. 4
plt.figure(figsize=(3.4, 2.8))
for dev in ("T4", "P100", "TPU"):
    pis = [p for p in off["bootstrap"][dev]["prediction_intervals"] if p["measured"] and not p["censored"]]
    m = np.array([p["measured"] for p in pis], float); pr = np.array([p["pred"] for p in pis], float)
    lo = pr - np.array([p["pi_lo"] for p in pis]); hi = np.array([p["pi_hi"] for p in pis]) - pr
    lab = {"T4": "Tesla T4", "P100": "Tesla P100", "TPU": "TPU v5e"}[dev]
    plt.errorbar(m, pr, yerr=[lo, hi], fmt=M[dev], color=C[dev], ms=4, capsize=2, elinewidth=0.6, label=lab)
lim = [40, 1500]
plt.plot(lim, lim, "k--", lw=0.6, label="Perfect agreement")
plt.xscale("log"); plt.yscale("log"); plt.xlim(lim); plt.ylim(lim)
plt.xlabel(r"Measured break-even $NP^*$ (individuals)"); plt.ylabel(r"Predicted $NP^*$ (individuals)")
plt.grid(alpha=0.3, which="both", lw=0.4); plt.legend(frameon=False)
save("fig4_predicted")

# ---------------------------------------------------------------- Fig. 5
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3))
cols = plt.cm.viridis(np.linspace(0, 0.85, len(TPUA["fusion"])))
for r, col in zip(TPUA["fusion"], cols):
    G = np.array(r["G"], float); tot = np.array(r["total_s"]) * 1e6
    lab = f"$NP$={r['NP']}, $D$={r['D']}"
    ax[0].plot(G, tot / G, "o-", ms=3, color=col, label=lab)
    ax[1].plot(G, tot, "o", ms=3, color=col, label=lab)
    ax[1].plot(G, r["c_sync_s"] * 1e6 + G * r["t_work_s"] * 1e6, "-", color=col, lw=0.6)
ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
ax[0].set_xlabel(r"Fusion factor $G$ (generations per dispatch)")
ax[0].set_ylabel(r"Per-generation time ($\mu$s)"); ax[0].set_title("(a) Per-generation cost", fontsize=8)
ax[1].set_xlabel(r"Fusion factor $G$ (generations per dispatch)")
ax[1].set_ylabel(r"Total time per dispatch ($\mu$s)"); ax[1].set_title(r"(b) Total time and fit of Eq. (9)", fontsize=8)
for a in ax:
    a.grid(alpha=0.3, which="both", lw=0.4)
ax[0].legend(frameon=False, fontsize=6)
save("fig5_fusion")
print("figures written; panel 3(b) from", "revision sweep" if used_rev else "three-dimension sweep")

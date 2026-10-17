"""Every manuscript table, built from the raw result records (no number is typed by hand).
Imported by export_tables.py. Paths: DATA_DIR (default data), RESULTS_DIR (default results)."""
import json, os
import numpy as np

U = os.environ.get("DATA_DIR", "data")
RES = os.environ.get("RESULTS_DIR", "results")
L = lambda n: json.load(open(os.path.join(U, n)))
T4, P100 = L("final_Tesla_T4.json"), L("final_Tesla_P100_PCIE_16GB.json")
TPU, TPUA = L("gate_tpu_TPU_v5_lite.json"), L("tpu_addendum.json")
ALG = {"T4": L("algos_Tesla_T4.json"), "P100": L("algos_Tesla_P100_PCIE_16GB.json")}
FIT = {"T4": L("fitness_Tesla_T4.json"), "P100": L("fitness_Tesla_P100_PCIE_16GB.json")}
REV1, REV2 = L("revision_Tesla_T4.json"), L("revision2_Tesla_T4.json")
OFF = json.load(open(os.path.join(RES, "offline_results.json")))

f2 = lambda x: f"{x:.2f}"
pct = lambda x: f"{x * 100:.1f}%"
spct = lambda x: f"{x * 100:+.1f}%".replace("-", "−")

TABLES = {}

# ---------------------------------------------------------------- Table: platforms
rows = []
for lab, d, spec in (("Tesla T4 (Turing, sm_75)", T4, 320.0), ("Tesla P100 (Pascal, sm_60)", P100, 732.0)):
    bw = d["bandwidth"]["gpu"]["plateau_bytes_per_s"] / 1e9
    rows.append([lab, "Xeon 2.00 GHz", "PyTorch 2.3.1 / CUDA 11.8", f"{bw:.1f}", f"{spec:.0f}", f"{bw / spec * 100:.0f}%"])
bw = TPUA["bandwidth"]["peak_bytes_per_s"] / 1e9
rows.append(["TPU v5e (1 of 8 chips)", "AMD EPYC 7B13", "JAX 0.10.2", f"{bw:.1f}", "819", f"{bw / 819 * 100:.0f}%"])
TABLES["platforms"] = dict(
    caption="Evaluation platforms. Achieved bandwidth is the sustained STREAM-triad plateau, not the vendor specification.",
    header=["Accelerator", "Host CPU", "Framework", "Achieved (GB/s)", "Specification (GB/s)", "Efficiency"],
    widths=[2300, 1500, 2150, 1150, 1250, 1000], rows=rows)

# ---------------------------------------------------------------- Table: memory share
B_MEAS = REV1["s3_traffic"]["acc_bytes_per_el"]          # 124.6 B/element, large-N slope (T4)
rows = []
ms = {(r["dev"], r["D"]): r for r in OFF["memory_share"]}
for D in (10, 30, 50, 100, 200):
    row = [str(D)]
    for dev in ("T4", "P100"):
        r = ms[(dev, D)]
        sh = r["share_119"] * B_MEAS / 118.73
        row += [f"{r['N']:,}", f"{r['t_gpu_us']:.1f}", f"{r['share_52'] * 100:.2f}%", f"{sh * 100:.2f}%"]
    rows.append(row)
TABLES["share"] = dict(
    caption="Memory traffic as a share of GPU generation time at the measured break-even, for the fused count "
            f"($B$ = 52) and the measured eager traffic ($B$ = {B_MEAS:.0f} bytes per element) of Table 2.",
    header=[["", ("Tesla T4", 4), ("Tesla P100", 4)],
            ["$D$", "$N^{*}$", "$t_{GPU}$ (μs)", "$B$ = 52", f"$B$ = {B_MEAS:.0f}", "$N^{*}$", "$t_{GPU}$ (μs)",
             "$B$ = 52", f"$B$ = {B_MEAS:.0f}"]],
    widths=[600, 1050, 1100, 950, 950, 1050, 1100, 950, 950], rows=rows)

# ---------------------------------------------------------------- Table: traffic
bwm = OFF["bandwidth_model"]
TABLES["traffic"] = dict(
    caption="Traffic per element per generation (bytes per element) estimated by independent methods.",
    header=["Method", "Work $N$ (elements)", "Tesla T4", "Tesla P100"],
    widths=[4300, 2100, 1300, 1300], align=["left", "center", "center", "center"],
    rows=[["Fused traversal count (13 traversals, assumed in Eq. (2))", "any", "52", "52"],
          ["Instrumented eager count (22 ATen operations, no reuse)", "1.5×10<sup>4</sup>–1.2×10<sup>6</sup>",
           "116–119", "116–119"],
          ["Fitted $b$ × achieved bandwidth (54-point grid)", "32–1.2×10<sup>6</sup>",
           f"{bwm['T4']['effective_B_gpu']:.1f}", f"{bwm['P100']['effective_B_gpu']:.1f}"],
          ["Large-$N$ time per element × bandwidth (revision run)", "4.2×10<sup>6</sup>–1.7×10<sup>7</sup>",
           f"{REV1['s3_traffic']['acc_bytes_per_el']:.0f}", "—"],
          ["Kernel time (CUPTI trace) × bandwidth (revision run)", "4.2×10<sup>6</sup>",
           f"{next(r for r in REV1["s1_trace"]["rows"] if r["NP"] == 4096 and r["D"] == 1024)["kernel_time_traffic_B_per_el"]:.0f}", "—"],
          ["Hardware DRAM counters (Nsight Compute, nvprof, CUPTI)", "4.2×10<sup>6</sup>", "access refused", "—"]],
    note="Revision measurements were made on the T4 only; the P100 was no longer offered by the platform.")

# ---------------------------------------------------------------- Table: cost model with CIs
rows = []
labels = {("T4", "host"): "T4 host CPU", ("T4", "acc"): "T4 GPU", ("P100", "host"): "P100 host CPU",
          ("P100", "acc"): "P100 GPU", ("TPU", "host"): "TPU host (NumPy)", ("TPU", "acc"): "TPU v5e"}
BWACC = {"T4": T4["bandwidth"]["gpu"]["plateau_bytes_per_s"] / 1e9,
         "P100": P100["bandwidth"]["gpu"]["plateau_bytes_per_s"] / 1e9,
         "TPU": TPUA["bandwidth"]["peak_bytes_per_s"] / 1e9}
for dev in ("T4", "P100", "TPU"):
    for side in ("host", "acc"):
        e = OFF["bootstrap"][dev][side]
        bbw = "—"
        if side == "acc":
            bw_ = BWACC[dev]
            bbw = f"{e['b_ns'] * bw_:.2f} [{e['b_ci_ns'][0] * bw_:.1f}, {e['b_ci_ns'][1] * bw_:.1f}]"
        bd = 2 if e["b_ns"] > 1 else 4
        ad = 1 if e["a_ns"] > 10 else 2
        rows.append([labels[(dev, side)],
                     f"{e['c_us']:.2f} [{e['c_ci_us'][0]:.1f}, {e['c_ci_us'][1]:.1f}]",
                     f"{e['a_ns']:.2f} [{e['a_ci_ns'][0]:.{ad}f}, {e['a_ci_ns'][1]:.{ad}f}]",
                     f"{e['b_ns']:.4f} [{e['b_ci_ns'][0]:.{bd}f}, {e['b_ci_ns'][1]:.{bd}f}]",
                     f"{e['r2']:.4f}", bbw])
TABLES["costmodel"] = dict(
    caption="Fitted cost model of Eq. (3) (NNLS, 54-point grid per device) with 95% pairs-bootstrap confidence intervals "
            "(5,000 resamples); the last column is the per-element product of Eq. (8) with the achieved bandwidth of Table 1.",
    header=["Platform", "$c$ (μs) [95% CI]", "$a$ (ns) [95% CI]", "$b$ (ns) [95% CI]", "$R^{2}$", "$b$ × BW (B/element)"],
    widths=[1450, 1800, 1800, 1950, 700, 1650], rows=rows)

# ---------------------------------------------------------------- Table: predicted vs measured
PI = {dev: {p["D"]: p for p in OFF["bootstrap"][dev]["prediction_intervals"]} for dev in ("T4", "P100", "TPU")}
rows = []
for D in (10, 30, 50, 100, 200, 500):
    row = [str(D)]
    for dev in ("T4", "P100", "TPU"):
        p = PI[dev][D]
        if p["censored"]:
            row += [f"<{p['measured']}", f"{p['pred']:.0f}", "cens."]
        else:
            err = (p["pred"] - p["measured"]) / p["measured"]
            row += [f"{p['measured']:,}", f"{p['pred']:,.0f} [{p['pi_lo']:,.0f}, {p['pi_hi']:,.0f}]", spct(err)]
    rows.append(row)
TABLES["breakeven"] = dict(
    caption="Measured and predicted break-even population $NP^{*}$ with 95% bootstrap prediction intervals in brackets "
            "(cens.: below the eager floor of 32).",
    header=[["", ("Tesla T4", 3), ("Tesla P100", 3), ("TPU v5e", 3)],
            ["$D$", "Measured", "Predicted [95% PI]", "Error", "Measured", "Predicted [95% PI]", "Error",
             "Measured", "Predicted [95% PI]", "Error"]],
    widths=[450, 800, 1450, 700, 800, 1450, 700, 800, 1450, 700], rows=rows)

# ---------------------------------------------------------------- Table: repeatability / sessions
def _cell(vals):
    mu = float(np.mean(vals))
    if len(vals) > 1:
        return f"{mu:,.0f} ({np.std(vals, ddof=1) / mu * 100:.1f}%)", mu
    return f"{mu:,.0f}", mu


def _session_orig(d):
    out = {}
    rep = {r["D"]: r["values"] for r in d["repeatability"]}
    for r in d["breakeven_eager"]:
        if r.get("np_star") and not r.get("censored"):
            out[r["D"]] = rep.get(r["D"], []) + [r["np_star"]]
    return out


def _session_rev(rows):
    out = {}
    for r in rows:
        if r.get("np_star") and not r.get("censored"):
            out.setdefault(r["D"], []).append(r["np_star"])
    return out


def _pred(fc, fa, D):
    return (fa["c_s"] - fc["c_s"]) / (fc["a_s"] - fa["a_s"] + (fc["b_s"] - fa["b_s"]) * D)


SESSIONS = []
for lab, d in (("T4, original", T4), ("P100, original", P100)):
    meas = _session_orig(d)
    errs = [abs(r["rel_error"]) for r in d["breakeven_eager"] if r.get("rel_error") is not None]
    SESSIONS.append((lab, "separate blocks", d["fit"]["cuda"]["c_s"] * 1e6, meas, float(np.median(errs))))
for lab, sec, mode in (("T4, revision 1", REV1["s5_high_reps"], "separate blocks"),
                       ("T4, revision 2", REV2["s9_calibration"], "interleaved")):
    meas = _session_rev(sec["breakeven"])
    errs = [abs(_pred(sec["fit_cpu"], sec["fit_acc"], D) - np.mean(v)) / np.mean(v) for D, v in meas.items()]
    SESSIONS.append((lab, mode, sec["fit_acc"]["c_s"] * 1e6, meas, float(np.median(errs))))
rows = []
for lab, mode, cg, meas, med in SESSIONS:
    rows.append([lab, mode, f"{cg:.1f}"] + [_cell(meas[D])[0] if D in meas else "—" for D in (10, 30, 50, 100, 200)]
                + [f"{med * 100:.1f}%"])
TABLES["repeat"] = dict(
    caption="Break-even population $NP^{*}$ of DE in the original runs and two revision sessions: mean of independent measurements "
            "(coefficient of variation in brackets), fitted GPU fixed cost $c_{gpu}$, how the cost grid was timed, and "
            "the median absolute error of the model fitted in the same session.",
    header=["Device, session", "Grid timing", "$c_{gpu}$ (μs)", "$D$ = 10", "$D$ = 30", "$D$ = 50", "$D$ = 100",
            "$D$ = 200", "Median error"],
    widths=[1350, 1150, 750, 1100, 1050, 950, 950, 900, 900], rows=rows)

# ---------------------------------------------------------------- Table: b x BW
rows = []
for dev, lab, bwv, spec in (("T4", "Tesla T4", T4["bandwidth"]["gpu"]["plateau_bytes_per_s"] / 1e9, 320),
                            ("P100", "Tesla P100", P100["bandwidth"]["gpu"]["plateau_bytes_per_s"] / 1e9, 732),
                            ("TPU", "TPU v5e", TPUA["bandwidth"]["peak_bytes_per_s"] / 1e9, 819)):
    e = OFF["bootstrap"][dev]["acc"]
    rows.append([lab, f"{bwv:.1f}", f"{bwv / spec * 100:.0f}%",
                 f"{e['b_ns']:.4f} [{e['b_ci_ns'][0]:.4f}, {e['b_ci_ns'][1]:.4f}]",
                 f"{e['b_ns'] * bwv:.2f} [{e['b_ci_ns'][0] * bwv:.1f}, {e['b_ci_ns'][1] * bwv:.1f}]"])
TABLES["bwlaw"] = dict(
    caption="Per-element coefficient against achieved bandwidth, with 95% bootstrap confidence intervals.",
    header=["Accelerator", "Achieved BW (GB/s)", "Efficiency", "$b$ (ns) [95% CI]", "$b$ × BW (B/element) [95% CI]"],
    widths=[1500, 1600, 1100, 2300, 2500], rows=rows)

# ---------------------------------------------------------------- Table: objective cost
rows = []
for dev in ("T4", "P100"):
    f = FIT[dev]
    base = f["fits"]["sphere"]
    for fn, nm in (("sphere", "Sphere"), ("rastrigin", "Rastrigin"), ("ackley", "Ackley"), ("expensive8", "Synthetic (8 iterations)")):
        ft = f["fits"][fn]
        be = next(r for r in f["breakeven"][fn] if r["D"] == 30)
        pa = base["dc"] / (base["da"] + ft["db"] * 30)
        row = [dev, nm, f"{f['fitness_cost'][fn]['cpu_ns']:.2f}", str(be["np_star"]),
               f"{pa:.0f} ({(pa - be['np_star']) / be['np_star'] * 100:+.0f}%)".replace("-", "−"), f"{ft['gpu']['c_s'] * 1e6:.0f}"]
        own = "—"
        if dev == "T4":
            b7 = next(x for x in REV1["s7_objectives"][fn]["breakeven"] if x["D"] == 30)
            pr, me = b7["np_star_predicted_own"], b7["np_star"]
            own = f"{pr:.0f} / {me} ({(pr - me) / me * 100:+.0f}%)".replace("-", "−")
        row.append(own)
        rows.append(row)
TABLES["fitness"] = dict(
    caption="Break-even population at $D$ = 30 against objective cost. The a priori prediction keeps the fixed and "
            "per-individual costs of Sphere and changes only Δ$b$; the last column refits Eq. (3) by NNLS for each objective in "
            "the revision run (T4) and compares the prediction with the break-even re-measured in that run.",
    header=["Device", "Objective", "Cost (ns/element)", "Measured $NP^{*}$", "A priori prediction (error)",
            "GPU $c_{gpu}$ (μs)", "Own fit: predicted / measured (error)"],
    widths=[800, 2000, 1100, 1100, 1600, 1100, 1650], rows=rows)

# ---------------------------------------------------------------- Table: algorithms
rows = []
for r in OFF["nnls_vs_ols"]:
    std = ALG[r["dev"]]["algorithms"][r["alg"]]["cpu"]
    rows.append([r["dev"], {"de": "DE", "pso": "PSO", "jde": "jDE", "cmaes": "CMA-ES"}[r["alg"]],
                 f"{std['standard']['r2']:.4f}", f"{std['r2_gain']:+.4f}", pct(r["ols"]["median_abs"]),
                 pct(r["nnls"]["median_abs"]), f"{r['ratio_ols_over_nnls']:.2f}×"])
TABLES["algos"] = dict(
    caption="Model generality across population methods: a controlled comparison of OLS and NNLS on identical cost grids "
            "and break-even measurements (median absolute prediction error over $D$ = 10, 30, 100).",
    header=["Device", "Algorithm", "Standard $R^{2}$", "Extended gain", "Error (OLS)", "Error (NNLS)", "OLS/NNLS"],
    widths=[900, 1300, 1400, 1400, 1300, 1300, 1200], rows=rows)

# ---------------------------------------------------------------- Table: graph capture
rows = []
for dev, d in (("T4, original", T4), ("P100, original", P100)):
    g, v = d["graph_costs"], d["graph_verification"]
    rows.append([dev, f"{g['c_gpu_eager_s'] * 1e6:.1f}", f"{g['c_gpu_graph_s'] * 1e6:.1f}", f"{g['reduction']:.2f}×",
                 f"{g['c_cpu_s'] * 1e6:.1f}", f"{v['p']:.3f}", "4 (smallest tested)"])
p8 = REV2["s8_graphed"]
pr = p8["probes"]
rows.append(["T4, revision 2", f"{pr['c_gpu_eager_s'] * 1e6:.1f}", f"{pr['c_gpu_graph_s'] * 1e6:.1f}",
             f"{pr['c_gpu_eager_s'] / pr['c_gpu_graph_s']:.2f}×", f"{pr['c_cpu_s'] * 1e6:.1f}", f"{p8['gate']['p']:.3f}",
             "1 (all four $D$)"])
TABLES["graph"] = dict(
    caption="Fixed cost under CUDA graph capture (direct probes at $NP$ = 8, $D$ = 1) and the smallest population at "
            "which the captured generation was faster than the host at $D$ = 10, 30, 100 and 500.",
    header=["Device, session", "Eager $c$ (μs)", "Graphed $c$ (μs)", "Reduction", "Host $c$ (μs)", "KS $p$",
            "GPU faster from $NP$ ="],
    widths=[1500, 1200, 1350, 1050, 1150, 850, 1900], rows=rows)

# ---------------------------------------------------------------- Table: fusion
rows = [[f"{r['NP']:,}", str(r["D"]), f"{r['n_elements']:,}", f"{r['c_sync_s'] * 1e6:.2f}", f"{r['t_work_s'] * 1e6:.2f}",
         f"{r['r2']:.5f}"] for r in TPUA["fusion"]]
TABLES["fusion"] = dict(
    caption="Fusion decomposition total($G$) = $c_{sync}$ + $G$·$t_{work}$ on TPU v5e ($G$ = 1–64).",
    header=["$NP$", "$D$", "$N$ (elements)", "$c_{sync}$ (μs)", "$t_{work}$ (μs)", "$R^{2}$"],
    widths=[1200, 900, 1600, 1600, 1600, 1300], rows=rows)

# ---------------------------------------------------------------- Table: energy
rows = []
for dev, d in (("T4", T4), ("P100", P100)):
    for r in d["energy"]["rows"]:
        g = r["cuda"]
        n = r["NP"] * r["D"]
        rows.append([dev, f"{r['NP']:,}", str(r["D"]), f"{g['total_j_per_gen'] * 1e3:.2f}",
                     f"{g['total_j_per_gen'] / n * 1e6:.4f}", f"{r['cpu']['gen_per_s']:.1f}", f"{g['gen_per_s']:.1f}"])
TABLES["energy"] = dict(
    caption="Accelerator energy per generation from direct power sampling (host energy is not exposed on the platform).",
    header=["Device", "$NP$", "$D$", "mJ/generation", "μJ/element", "Host gen/s", "GPU gen/s"],
    widths=[900, 1100, 800, 1500, 1400, 1400, 1400], rows=rows)

# ---------------------------------------------------------------- Table: comparison
TABLES["compare"] = dict(
    allow_split=True,
    caption="Comparison with existing systems and recent studies.",
    header=["Study", "Setting", "Main finding", "Crossover mechanism", "Predicts break-even"],
    widths=[2350, 1950, 2700, 1450, 900], align=["left", "left", "left", "left", "left"],
    rows=[["Frameworks [@evox;@evotorch;@evosax], tensorized MOEAs [@liang24;@liang26;@li25]", "GPU evolutionary computation",
           "Speedups at large scale", "Not analysed", "No"],
          ["Dickson et al. [@dickson]", "Optimized Monte Carlo code", "Vectorized CPU about 2× faster than GPU",
           "Vectorization", "No"],
          ["Syberfeldt and Ekblom [@syberfeldt]", "Independent EA runs", "CPU faster in some scenarios", "Not modelled",
           "Guidance"],
          ["Eynaliyev and Liu [@eynaliyev]", "Simulation-based EC", "CPU generally faster", "Not modelled", "No"],
          ["Yu et al. [@yu26]", "16 EAs, 30 problems, two GPUs", "Under-utilized, effective, saturated regimes",
           "Resource management", "No"],
          ["Ekelund et al. [@ekelund]", "Iterative solvers", "CUDA-graph batching gives >1.4×", "Launch overhead",
           "Batch size"],
          ["This work", "Four EAs; T4, P100, TPU", "Fixed synchronization cost; reducible", "Three-term model",
           "Yes (10–20%)"]])

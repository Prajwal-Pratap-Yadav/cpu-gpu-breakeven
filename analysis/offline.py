#!/usr/bin/env python3
"""Offline analyses for the revision (paper 2095). Reads the original result JSONs
only; writes offline_results.json. Every number used in the response letter that
does not come from the new Kaggle run is produced here."""
import json, os, sys
import numpy as np
from scipy.optimize import nnls
from scipy import stats

IN = sys.argv[1] if len(sys.argv) > 1 else "data"
OUTF = sys.argv[2] if len(sys.argv) > 2 else "results/offline_results.json"
OUT = {}
RNG = np.random.default_rng(20260820)
B = 5000


def load(n):
    with open(os.path.join(IN, n)) as fh:
        return json.load(fh)


T4, P100 = load("final_Tesla_T4.json"), load("final_Tesla_P100_PCIE_16GB.json")
TPU, TPUA = load("gate_tpu_TPU_v5_lite.json"), load("tpu_addendum.json")
ALG = {"T4": load("algos_Tesla_T4.json"), "P100": load("algos_Tesla_P100_PCIE_16GB.json")}
FIT = {"T4": load("fitness_Tesla_T4.json"), "P100": load("fitness_Tesla_P100_PCIE_16GB.json")}
G3 = {}


def design(rows):
    NP = np.array([r["NP"] for r in rows], float)
    D = np.array([r["D"] for r in rows], float)
    t = np.array([r["time_s"] for r in rows], float)
    return np.column_stack([np.ones_like(NP), NP, NP * D]), t


def fit(A, t, constrained=True):
    if constrained:
        coef, _ = nnls(A, t)
    else:
        coef = np.linalg.lstsq(A, t, rcond=None)[0]
    pred = A @ coef
    r2 = 1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    return coef, r2


def boot(A, t):
    n = len(t)
    out = np.empty((B, 3))
    for i in range(B):
        idx = RNG.integers(0, n, n)
        out[i] = fit(A[idx], t[idx])[0]
    return out


def np_star(cc, ca, D):
    dc = ca[0] - cc[0]
    den = (cc[1] - ca[1]) + (cc[2] - ca[2]) * D
    return dc / den if (dc > 0 and den > 0) else np.nan


# ------------------------------------------------ A. bootstrap CIs (R1.3)
grids = {
    "T4": (T4["grid"]["cpu"], T4["grid"]["cuda"]),
    "P100": (P100["grid"]["cpu"], P100["grid"]["cuda"]),
    "TPU": (TPU["grid"]["cpu"], TPU["grid"]["acc"]),
}
measured = {
    "T4": T4["breakeven_eager"], "P100": P100["breakeven_eager"], "TPU": TPU["breakeven"],
}
ci = {}
for dev, (gc, ga) in grids.items():
    Ac, tc = design(gc)
    Aa, ta = design(ga)
    cc, r2c = fit(Ac, tc)
    ca, r2a = fit(Aa, ta)
    bc, ba = boot(Ac, tc), boot(Aa, ta)
    ent = {}
    for side, coef, r2, bs in (("host", cc, r2c, bc), ("acc", ca, r2a, ba)):
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        ent[side] = {"c_us": coef[0] * 1e6, "a_ns": coef[1] * 1e9, "b_ns": coef[2] * 1e9, "r2": r2,
                     "c_ci_us": [lo[0] * 1e6, hi[0] * 1e6], "a_ci_ns": [lo[1] * 1e9, hi[1] * 1e9],
                     "b_ci_ns": [lo[2] * 1e9, hi[2] * 1e9],
                     "frac_a_zero": float((bs[:, 1] <= 1e-15).mean()),
                     "frac_b_zero": float((bs[:, 2] <= 1e-18).mean()),
                     "a_ci_rel_halfwidth": float((hi[1] - lo[1]) / 2 / coef[1]) if coef[1] > 0 else None,
                     "b_ci_rel_halfwidth": float((hi[2] - lo[2]) / 2 / coef[2]) if coef[2] > 0 else None,
                     "c_ci_rel_halfwidth": float((hi[0] - lo[0]) / 2 / coef[0])}
    # share of delta-a and delta-b from the host
    ent["da_host_share"] = cc[1] / (cc[1] - ca[1]) if cc[1] - ca[1] > 0 else None
    ent["db_host_share"] = cc[2] / (cc[2] - ca[2])
    # prediction intervals for NP*
    pis = []
    for r in measured[dev]:
        D = r["D"]
        draws = np.array([np_star(bc[i], ba[i], D) for i in range(B)])
        draws = draws[np.isfinite(draws)]
        p = np_star(cc, ca, D)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        m = r.get("np_star")
        cen = r.get("censored", False)
        pis.append({"D": D, "pred": p, "pi_lo": lo, "pi_hi": hi, "measured": m, "censored": cen,
                    "pi_rel_halfwidth": (hi - lo) / 2 / p,
                    "measured_inside": (lo <= m <= hi) if (m and not cen) else None})
    ent["prediction_intervals"] = pis
    ci[dev] = ent
OUT["bootstrap"] = ci

# ------------------------------------------------ B. signature (R1.4)
def signature(rows):
    pts = [(r["D"], r["np_star"]) for r in rows if r.get("np_star") and not r.get("censored")]
    if len(pts) < 3:
        return {"n": len(pts)}
    D = np.array([p[0] for p in pts], float)
    inv = 1 / np.array([p[1] for p in pts], float)
    res = stats.linregress(D, inv)
    return {"n": len(pts), "D": D.tolist(), "np": [p[1] for p in pts], "slope": res.slope,
            "intercept": res.intercept, "r2": res.rvalue ** 2, "p_slope": res.pvalue,
            "slope_se": res.stderr, "intercept_se": res.intercept_stderr,
            "D_range": float(D.max() / D.min())}


sig = {"DE_main": {k: signature(v) for k, v in measured.items()}}
sig["algorithms"] = {dev: {al: signature(ALG[dev]["algorithms"][al]["breakeven"])
                           for al in ALG[dev]["algorithms"]} for dev in ALG}
sig["objectives"] = {dev: {fn: signature(FIT[dev]["breakeven"][fn]) for fn in FIT[dev]["breakeven"]}
                     for dev in FIT}
OUT["signature"] = sig

# ------------------------------------------------ C. NNLS vs OLS (R1.5)
tab8 = []
for dev in ALG:
    for al, A_ in ALG[dev]["algorithms"].items():
        Ac, tc = design(A_["cpu"]["grid"])
        Aa, ta = design(A_["cuda"]["grid"])
        res = {}
        for est, con in (("ols", False), ("nnls", True)):
            cc, r2c = fit(Ac, tc, con)
            ca, r2a = fit(Aa, ta, con)
            errs = []
            for r in A_["breakeven"]:
                if r.get("np_star") and not r.get("censored"):
                    p = np_star(cc, ca, r["D"])
                    errs.append((p - r["np_star"]) / r["np_star"] if np.isfinite(p) else np.nan)
            res[est] = {"a_cpu_ns": cc[1] * 1e9, "a_acc_ns": ca[1] * 1e9, "c_cpu_us": cc[0] * 1e6,
                        "c_acc_us": ca[0] * 1e6, "r2_cpu": r2c, "r2_acc": r2a, "errors": errs,
                        "median_abs": float(np.nanmedian(np.abs(errs))),
                        "max_abs": float(np.nanmax(np.abs(errs)))}
        tab8.append({"dev": dev, "alg": al, "ols": res["ols"], "nnls": res["nnls"],
                     "ratio_ols_over_nnls": res["ols"]["median_abs"] / res["nnls"]["median_abs"],
                     "dims": [r["D"] for r in A_["breakeven"]],
                     "measured": [r.get("np_star") for r in A_["breakeven"]]})
OUT["nnls_vs_ols"] = tab8

# ------------------------------------------------ D. fixed-cost reconciliation (R2.1)
fx = {}
for dev, d in (("T4", T4), ("P100", P100)):
    fx[dev] = {"gpu_fit_us": d["fit"]["cuda"]["c_s"] * 1e6,
               "gpu_probe_us": d["graph_costs"]["c_gpu_eager_s"] * 1e6,
               "cpu_fit_us": d["fit"]["cpu"]["c_s"] * 1e6,
               "cpu_probe_us": d["graph_costs"]["c_cpu_s"] * 1e6,
               "gpu_graph_us": d["graph_costs"]["c_gpu_graph_s"] * 1e6}
    fx[dev]["gpu_fit_over_probe"] = fx[dev]["gpu_fit_us"] / fx[dev]["gpu_probe_us"]
    fx[dev]["cpu_fit_over_probe"] = fx[dev]["cpu_fit_us"] / fx[dev]["cpu_probe_us"]
    g3 = G3.get(dev, {})
    fx[dev]["gate3_c_direct"] = {k: v for k, v in g3.get("c_direct", {}).items()
                                 if not isinstance(v, (dict, list))}
fx["TPU"] = {"acc_fit_us": TPU["fit"]["acc"]["c_s"] * 1e6, "acc_probe_us": TPU["c_direct"]["acc_s"] * 1e6,
             "cpu_fit_us": TPU["fit"]["cpu"]["c_s"] * 1e6, "cpu_probe_us": TPU["c_direct"]["cpu_s"] * 1e6}
OUT["fixed_cost"] = fx

# ------------------------------------------------ E. bandwidth model (R1.2)
def nbw(dc, bwc, bwg, Bb):
    return dc * bwc * bwg / (Bb * (bwg - bwc))


bwm = {}
for dev, d in (("T4", T4), ("P100", P100)):
    bwc_pl = d["bandwidth"]["cpu"]["plateau_bytes_per_s"]
    bwc_pk = d["bandwidth"]["cpu"]["peak_bytes_per_s"]
    bwg = d["bandwidth"]["gpu"]["plateau_bytes_per_s"]
    meas = [r["n_star"] for r in d["breakeven_eager"] if r.get("np_star") and not r.get("censored")]
    mmean = float(np.mean(meas))
    dc_probe = (d["graph_costs"]["c_gpu_eager_s"] - d["graph_costs"]["c_cpu_s"])
    dc_fit = d["fit"]["cuda"]["c_s"] - d["fit"]["cpu"]["c_s"]
    rows = []
    for bname, Bb in (("fused_52", 52.0), ("unfused_116", 116.27), ("unfused_119", 118.73)):
        for bwname, bwc in (("plateau", bwc_pl), ("peak", bwc_pk)):
            for cname, dc in (("probe", dc_probe), ("fit", dc_fit)):
                n = nbw(dc, bwc, bwg, Bb)
                rows.append({"B": bname, "host_bw": bwname, "dc": cname, "n_star_bw": n,
                             "overestimate": n / mmean})
    # B needed for agreement
    req = {bwname: dc_probe * bwc * bwg / ((bwg - bwc) * mmean)
           for bwname, bwc in (("plateau", bwc_pl), ("peak", bwc_pk))}
    bwm[dev] = {"measured_mean_n_star": mmean, "measured": meas, "bw_cpu_plateau": bwc_pl,
                "bw_cpu_peak": bwc_pk, "bw_gpu": bwg, "dc_probe_s": dc_probe, "dc_fit_s": dc_fit,
                "variants": rows, "B_required": req,
                "effective_B_gpu": d["fit"]["cuda"]["b_s"] * bwg,
                "effective_B_cpu_plateau": d["fit"]["cpu"]["b_s"] * bwc_pl,
                "effective_B_cpu_peak": d["fit"]["cpu"]["b_s"] * bwc_pk}
bwm["TPU"] = {"effective_B_acc": TPU["fit"]["acc"]["b_s_per_element"] * TPUA["bandwidth"]["peak_bytes_per_s"]}
OUT["bandwidth_model"] = bwm

# memory share at the break-even (Table 2), three traffic estimates
share = []
for dev, d in (("T4", T4), ("P100", P100)):
    bwg = d["bandwidth"]["gpu"]["plateau_bytes_per_s"]
    for r in d["breakeven_eager"]:
        if not r.get("np_star") or r.get("censored"):
            continue
        N = r["n_star"]
        tg = r["t_gpu_s"]
        row = {"dev": dev, "D": r["D"], "N": N, "t_gpu_us": tg * 1e6}
        for nm, Bb in (("52", 52.0), ("119", 118.73)):
            row[f"mem_us_{nm}"] = N * Bb / bwg * 1e6
            row[f"share_{nm}"] = N * Bb / bwg / tg
        row["mem_us_fit"] = d["fit"]["cuda"]["b_s"] * N * 1e6
        row["share_fit"] = d["fit"]["cuda"]["b_s"] * N / tg
        share.append(row)
OUT["memory_share"] = share
OUT["memory_share_range"] = {k: [min(r[k] for r in share), max(r[k] for r in share)]
                             for k in ("share_52", "share_119", "share_fit")}

# GPU flatness (Fig. 2 statistics)
flat = {}
for dev, d in (("T4", T4), ("P100", P100)):
    g = d["grid"]["cuda"]
    t = np.array([r["time_s"] for r in g]) * 1e6
    c = np.array([r["time_s"] for r in d["grid"]["cpu"]]) * 1e6
    n = np.array([r["NP"] * r["D"] for r in g])
    flat[dev] = {"gpu_mean_us": t.mean(), "gpu_cv": t.std(ddof=1) / t.mean(),
                 "gpu_cv_excl_max": np.delete(t, t.argmax()).std(ddof=1) / np.delete(t, t.argmax()).mean(),
                 "gpu_max_us": t.max(), "argmax": g[int(t.argmax())],
                 "cpu_ratio_max_min": c.max() / c.min(), "N_min": int(n.min()), "N_max": int(n.max())}
OUT["flatness"] = flat

# ------------------------------------------------ F. error structure (R2.4)
es = {}
for dev, d in (("T4", T4), ("P100", P100), ("TPU", TPU)):
    rows = [r for r in (d["breakeven_eager"] if dev != "TPU" else d["breakeven"])
            if r.get("rel_error") is not None and not r.get("censored")]
    D = [r["D"] for r in rows]
    e = [r["rel_error"] for r in rows]
    rho = stats.spearmanr(D, e)
    es[dev] = {"D": D, "err": e, "spearman": rho.statistic, "p": rho.pvalue,
               "median_abs": float(np.median(np.abs(e))), "mean_signed": float(np.mean(e))}
    if dev != "TPU":
        pooled = []
        for rep in d["repeatability"]:
            main = next(r for r in d["breakeven_eager"] if r["D"] == rep["D"])
            vals = rep["values"] + [main["np_star"]]
            mu = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            pred = main["np_star_predicted"]
            pooled.append({"D": rep["D"], "values": vals, "mean": mu, "cv": float(np.std(vals, ddof=1) / mu),
                           "sem_rel": sem / mu, "pred": pred, "err_vs_pooled": (pred - mu) / mu,
                           "z": (pred - mu) / sem})
        es[dev]["pooled"] = pooled
all_err = [abs(x) for k in ("T4", "P100") for x in es[k]["err"]]
OUT["error_structure"] = es
OUT["error_summary"] = {"median_abs_gpu": float(np.median(all_err)), "max_abs_gpu": float(max(all_err)),
                        "n": len(all_err)}

# ------------------------------------------------ G. objective cost (R2.5)
obj = {}
for dev in FIT:
    f = FIT[dev]
    base = f["fits"]["sphere"]
    rows = []
    for fn in f["fits"]:
        ft = f["fits"][fn]
        be30 = next((r for r in f["breakeven"][fn] if r["D"] == 30), None)
        # a-priori prediction: objective enters only b (sphere's dc and da kept)
        pa = base["dc"] / (base["da"] + ft["db"] * 30)
        rows.append({"fn": fn, "cost_ns": f["fitness_cost"][fn]["cpu_ns"],
                     "dc_us": ft["dc"] * 1e6, "da_ns": ft["da"] * 1e9, "db_ns": ft["db"] * 1e9,
                     "dc_ratio": ft["dc"] / base["dc"], "db_ratio": ft["db"] / base["db"],
                     "np30": be30["np_star"] if be30 else None,
                     "pred_own_params": be30.get("np_star_predicted") if be30 else None,
                     "err_own_params": be30.get("rel_error") if be30 else None,
                     "pred_apriori": pa,
                     "err_apriori": (pa - be30["np_star"]) / be30["np_star"] if be30 else None,
                     "usable": ft.get("usable"), "flags": ft.get("flags"),
                     "c_cpu_us": ft["cpu"]["c_s"] * 1e6, "c_gpu_us": ft["gpu"]["c_s"] * 1e6,
                     "errs_all_dims": [r.get("rel_error") for r in f["breakeven"][fn]]})
    x = np.log([r["cost_ns"] for r in rows])
    y_meas = np.log([r["np30"] for r in rows])
    y_ap = np.log([r["pred_apriori"] for r in rows])
    obj[dev] = {"rows": rows, "slope_measured": float(np.polyfit(x, y_meas, 1)[0]),
                "slope_apriori": float(np.polyfit(x, y_ap, 1)[0]), "trend": f["trend"]}
OUT["objective_cost"] = obj

# ------------------------------------------------ H. TPU fusion summary for Fig. 5
OUT["tpu_fusion"] = {"summary": TPUA["fusion_summary"],
                     "configs": [{k: r[k] for k in ("NP", "D", "n_elements", "c_sync_s", "t_work_s", "r2")}
                                 for r in TPUA["fusion"]]}
OUT["bandwidth_table"] = {
    "T4": T4["bandwidth"]["gpu"]["plateau_bytes_per_s"] / 1e9,
    "P100": P100["bandwidth"]["gpu"]["plateau_bytes_per_s"] / 1e9,
    "TPU": TPUA["bandwidth"]["peak_bytes_per_s"] / 1e9}

with open(OUTF, "w") as fh:
    json.dump(OUT, fh, indent=1, default=float)
print("done")

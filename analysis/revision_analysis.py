"""Numbers quoted in the text from the two revision sessions on the Tesla T4
(Sections 4.4, 4.7, 5.1, 5.3, 5.5, 5.7 and the responses to the reviewers).
Usage: python3 analysis/revision_analysis.py [data_dir] [out_json]"""
import json, sys
import numpy as np

U = sys.argv[1] if len(sys.argv) > 1 else "data"
OUTF = sys.argv[2] if len(sys.argv) > 2 else "results/revision_summary.json"
R1 = json.load(open(f"{U}/revision_Tesla_T4.json"))
R2 = json.load(open(f"{U}/revision2_Tesla_T4.json"))
out = {}


def nstar(fc, fa, D):
    return (fa["c_s"] - fc["c_s"]) / (fc["a_s"] - fa["a_s"] + (fc["b_s"] - fa["b_s"]) * D)


def by_D(rows):
    d = {}
    for r in rows:
        if r.get("np_star") and not r.get("censored"):
            d.setdefault(r["D"], []).append(r["np_star"])
    return d


# S1: kernel trace -- how long the GPU is busy near the break-even (NP = 128-512, D = 30-100)
near = [r for r in R1["s1_trace"]["rows"] if 128 <= r["NP"] <= 512]
out["s1_near_breakeven"] = {"kernels_per_gen": sorted({r["kernels_per_gen"] for r in near}),
                            "busy_us": [min(r["busy_us"] for r in near), max(r["busy_us"] for r in near)],
                            "wall_us": [min(r["wall_us"] for r in near), max(r["wall_us"] for r in near)],
                            "busy_fraction": [min(r["busy_fraction"] for r in near), max(r["busy_fraction"] for r in near)]}
big = next(r for r in R1["s1_trace"]["rows"] if r["NP"] == 4096 and r["D"] == 1024)
out["s1_kernel_time_traffic_B_per_el"] = big["kernel_time_traffic_B_per_el"]

# S1: GPU time per generation against the number of kernels of each objective (NP = 256, D = 30)
ob = R1["s1_objectives"]
k = np.array([o["kernels_per_gen"] for o in ob], float); w = np.array([o["wall_us"] for o in ob], float)
sl, ic = np.polyfit(k, w, 1)
r2 = 1 - ((w - (sl * k + ic)) ** 2).sum() / ((w - w.mean()) ** 2).sum()
out["s1_objectives"] = {"kernels": dict(zip([o["objective"] for o in ob], k.tolist())),
                        "wall_us": [w.min(), w.max()], "busy_us": [min(o["busy_us"] for o in ob), max(o["busy_us"] for o in ob)],
                        "us_per_kernel": sl, "r2": r2}

# S2/S3: traffic
out["s2_counters_refused"] = {t: R1["s2_counters"][t]["refused"] for t in ("ncu", "nvprof")} | \
                             {"cupti_counter_values": R1["s2_counters"]["cupti_metrics"]["counter_values_found"]}
out["s3_bytes_per_element"] = {"gpu": R1["s3_traffic"]["acc_bytes_per_el"], "host": R1["s3_traffic"]["cpu_bytes_per_el"]}

# S4: captured generation against the host at NP = 1..16 (revision 1)
rows = R1["s4_small_np"]["rows"]
host_faster = [r for r in rows if not r["gpu_faster"]]
out["s4_graph_vs_host"] = {"points": len(rows), "host_faster": len(host_faster),
                           "max_host_advantage_D_le_30": max(1 / r["ratio_cpu_over_graph"] - 1 for r in rows if r["D"] <= 30),
                           "ks_p": R1["s4_small_np"]["gate"]["p"]}

# S5 (revision 1, separate blocks) and S9 (revision 2, interleaved): sessions
for key, sec in (("revision1_separate_blocks", R1["s5_high_reps"]), ("revision2_interleaved", R2["s9_calibration"])):
    meas = by_D(sec["breakeven"])
    cv = {D: float(np.std(v, ddof=1) / np.mean(v)) for D, v in meas.items() if len(v) > 1}
    err = {D: float((nstar(sec["fit_cpu"], sec["fit_acc"], D) - np.mean(v)) / np.mean(v)) for D, v in meas.items()}
    out[key] = {"c_gpu_fit_us": sec["fit_acc"]["c_s"] * 1e6, "c_cpu_fit_us": sec["fit_cpu"]["c_s"] * 1e6,
                "breakeven_mean": {D: float(np.mean(v)) for D, v in meas.items()}, "cv": cv,
                "cv_range": [min(cv.values()), max(cv.values())], "rel_error": err,
                "abs_error_range": [min(map(abs, err.values())), max(map(abs, err.values()))],
                "median_abs_error": float(np.median([abs(e) for e in err.values()])),
                "steal_fraction": sec["steal_fraction"]}
ps, pe = R2["s9_calibration"]["probes_start"], R2["s9_calibration"]["probes_end"]
out["revision2_drift_within_session"] = {k_: (pe[k_] - ps[k_]) / ps[k_] for k_ in ps}

# S8: break-even under CUDA graph capture (revision 2)
p8 = R2["s8_graphed"]
out["s8_graphed"] = {"probes_us": {k_: v * 1e6 for k_, v in p8["probes"].items()},
                     "reduction": p8["probes"]["c_gpu_eager_s"] / p8["probes"]["c_gpu_graph_s"],
                     "graphed_np_star": {r["D"]: r["note"] for r in p8["rows"]},
                     "eager_np_star_same_session": {r["D"]: (("<" if r["eager"]["censored"] else "") + str(r["eager"]["np_star"]))
                                                    for r in p8["rows"]},
                     "ks_p": p8["gate"]["p"]}

# S7: objective sweep -- re-measured break-even at D = 30 and per-objective own fit
s7 = {}
for fn, v in R1["s7_objectives"].items():
    b = next(x for x in v["breakeven"] if x["D"] == 30)
    s7[fn] = {"np_star_D30": b["np_star"], "own_fit_pred": b["np_star_predicted_own"],
              "own_fit_error": (b["np_star_predicted_own"] - b["np_star"]) / b["np_star"]}
out["s7_objectives"] = s7

# S6: six-dimension signatures
out["s6_signature_r2"] = {al: v["signature_r2"] for al, v in R1["s6_signature"].items()}
out["steal_max"] = max(list(R1["section_steal"].values()) + list(R2["section_steal"].values()))


def show(d, ind=""):
    for k_, v in d.items():
        if isinstance(v, dict):
            print(f"{ind}{k_}:"); show(v, ind + "  ")
        else:
            print(f"{ind}{k_}: {np.round(v, 4).tolist() if isinstance(v, (list, float)) else v}")


show(out)
json.dump(out, open(OUTF, "w"), indent=1, default=float)

#!/usr/bin/env python3
"""
================================================================================
 EXPERIMENT: DOES THE BREAK-EVEN SCALE WITH FITNESS EVALUATION COST?
================================================================================
 THE OBJECTION THIS ANSWERS

 Every result so far used Sphere -- sum(x^2), the cheapest fitness function
 that exists. A reviewer will write: "the authors conclude overhead dominates,
 having chosen the workload with the least possible work." That objection is
 correct and fatal if unanswered.

 THE PREDICTION, STATED BEFORE MEASURING

 Fitness cost enters the model through b (per-element). Since

     N*(D) = dc / (da + db*D),      db = b_cpu - b_gpu

 raising fitness cost raises db and therefore SHRINKS the break-even. Cheap
 functions should be overhead-dominated with a high N*; expensive functions
 should be work-dominated with a low N*. Concretely, over the 16x cost range
 measured here, N* at D=30 should fall by roughly an order of magnitude.

 THE SHARP TEST

 We do not merely check the trend. We fit dc and da on SPHERE ONLY, then
 predict every other function's break-even using only that function's
 independently measured b. Each non-Sphere function is therefore an
 out-of-sample prediction across fitness cost -- the same discipline used for
 the cross-architecture test.

 WHAT WOULD FALSIFY IT
   - N* flat across a 16x cost range -> b is not where fitness cost enters,
     and the model's decomposition is wrong.
   - N* falling but not as 1/(a + b*D) -> the functional form is wrong.
   Either outcome is reportable and must not be buried.

 KAGGLE: pin torch==2.3.1+cu118 (see KAGGLE_SETUP.md), restart, then run.
         Writes fitness_<gpu>.json.
================================================================================
"""
import json, math, os, platform, sys, time
from datetime import datetime, timezone

import numpy as np
import torch

SEED        = 20260820
DTYPE       = torch.float32
# v1 used a 4x4 = 16-point grid to save runtime. That was too small: fitting
# three parameters to 16 noisy points produced NEGATIVE c_cpu (-967.7 us on
# P100/expensive8), NEGATIVE b_gpu (ackley, R2=0.147) and NEGATIVE dc
# (T4/schwefel). The giveaway was that each function's OWN fitted parameters
# predicted its break-even WORSE than Sphere's did (28.1% vs 14.4% median,
# worst case +1329%) -- the signature of fitting noise. Back to the 54-point
# grid used for the cross-architecture work, on a reduced function set that
# still spans the full cost range.
GRID_NP     = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
GRID_D      = [2, 5, 10, 30, 100, 300]
BE_DIMS     = [10, 30, 100]
NP_FLOOR    = 32
REPS, WARM  = 15, 5
C_PROBE_NP, C_PROBE_D, C_PROBE_REPS = 8, 1, 101
CONFIRM_MARGIN  = 0.08   # within 8% -> re-time before believing the sign
CONFIRM_ROUNDS  = 3
MEM_FRACTION, TEMPS_PER_GEN = 0.20, 12

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================ fitness suite
def f_sphere(z):
    return (z * z).sum(dim=1)


def f_rastrigin(z):
    return 10 * z.shape[1] + (z * z - 10 * torch.cos(2 * math.pi * z)).sum(dim=1)


def f_schwefel(z):
    return 418.9829 * z.shape[1] - (z * torch.sin(torch.sqrt(torch.abs(z)))).sum(dim=1)


def f_ackley(z):
    d = z.shape[1]
    return (-20 * torch.exp(-0.2 * torch.sqrt((z * z).sum(dim=1) / d))
            - torch.exp(torch.cos(2 * math.pi * z).sum(dim=1) / d) + 20 + math.e)


def f_rosenbrock(z):
    return (100 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (z[:, :-1] - 1) ** 2).sum(dim=1)


def f_griewank(z):
    d = z.shape[1]
    i = torch.arange(1, d + 1, device=z.device, dtype=z.dtype)
    return (1 + (z * z).sum(dim=1) / 4000
            - torch.prod(torch.cos(z / torch.sqrt(i)), dim=1))


def f_expensive(z):
    """Deliberately costly synthetic: extends the range past any standard
    benchmark so the prediction is tested over 16x rather than 8x."""
    y = z
    for _ in range(8):
        y = torch.sin(y) * torch.cos(y) + torch.sqrt(torch.abs(y) + 1e-9)
    return (y * y).sum(dim=1)


# Four functions spanning the same ~40x cost range as the original seven.
# Fewer functions x a 3.4x larger grid is the right trade: the failure was
# never coverage, it was parameter identifiability.
FUNCTIONS = [("sphere", f_sphere),          # ~1.0 ns/el   cheapest possible
             ("rastrigin", f_rastrigin),    # ~4.5 ns/el   transcendentals
             ("ackley", f_ackley),          # ~9.1 ns/el   sqrt + exp + cos
             ("expensive8", f_expensive)]   # ~43  ns/el   8 fused iterations


# =================================================================== timing
def _sum(s):
    a = np.asarray(s, float)
    return {"median_s": float(np.median(a)), "mean_s": float(a.mean()),
            "std_s": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "p05_s": float(np.percentile(a, 5)),
            "p95_s": float(np.percentile(a, 95)), "n": int(a.size)}


def time_fn(fn, cuda, warmup=WARM, reps=REPS):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return _sum(out)


def de_generation(x, evaluate):
    NP, D = x.shape
    dev = x.device
    idx = torch.randint(0, NP, (3, NP), device=dev)
    v = x[idx[0]] + 0.5 * (x[idx[1]] - x[idx[2]])
    mask = torch.rand(NP, D, device=dev) < 0.9
    mask.scatter_(1, torch.randint(0, D, (NP, 1), device=dev), True)
    u = torch.where(mask, v, x)
    better = (evaluate(u) < evaluate(x)).unsqueeze(1)
    return torch.where(better, u, x)


def max_np_for(D, device):
    if device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        budget = free * MEM_FRACTION
    else:
        try:
            with open("/proc/meminfo") as fh:
                budget = next(int(l.split()[1]) * 1024 for l in fh
                              if l.startswith("MemAvailable")) * MEM_FRACTION
        except Exception:
            budget = (4 << 30) * MEM_FRACTION
    return max(8, int(budget / (TEMPS_PER_GEN * D * DTYPE_BYTES())))


def DTYPE_BYTES():
    return 4


def time_generation(device, NP, D, evaluate, reps=REPS, warmup=WARM):
    cuda = device == "cuda"
    x = torch.rand(NP, D, dtype=DTYPE, device=torch.device(device))
    st = time_fn(lambda: de_generation(x, evaluate), cuda, warmup, reps)
    del x
    if cuda:
        torch.cuda.empty_cache()
    return st["median_s"], st


# ============================== (1) isolated fitness cost, per function
def fitness_cost(device, evaluate, NP=512, D=30):
    """Cost of the fitness call alone, independent of the DE machinery."""
    cuda = device == "cuda"
    x = torch.rand(NP, D, dtype=DTYPE, device=torch.device(device))
    st = time_fn(lambda: evaluate(x), cuda, warmup=10, reps=31)
    del x
    if cuda:
        torch.cuda.empty_cache()
    return st["median_s"] / (NP * D), st


# ============================== (2) cost grid and fit, per function
def cost_grid(device, evaluate):
    rows = []
    for D in GRID_D:
        for NP in GRID_NP:
            if NP > max_np_for(D, device):
                continue
            try:
                t, st = time_generation(device, NP, D, evaluate, reps=11)
            except RuntimeError:
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            rows.append({"NP": NP, "D": D, "time_s": t})
    return rows


def fit_full(rows):
    NP = np.array([r["NP"] for r in rows], float)
    D = np.array([r["D"] for r in rows], float)
    t = np.array([r["time_s"] for r in rows], float)
    A = np.column_stack([np.ones_like(NP), NP, NP * D])
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    pred = A @ coef
    ss_res = float(((t - pred) ** 2).sum()); ss_tot = float(((t - t.mean()) ** 2).sum())
    return {"c_s": float(coef[0]), "a_s": float(coef[1]), "b_s": float(coef[2]),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "n_points": len(rows)}


# ============================== (3) measured break-even, per function
def measure_breakeven(D, evaluate, np_floor=NP_FLOOR):
    hi = min(max_np_for(D, "cuda"), max_np_for(D, "cpu"), 2 ** 20)
    if hi <= np_floor:
        return {"D": D, "np_star": None, "bracket": "no headroom"}

    def wins(NP, confirm=False):
        """
        Near the crossover the CPU and GPU curves meet at a shallow angle, so a
        single timing pair can flip sign on noise. v1 reported Sphere D=10 with
        t_cpu=320.0us and t_gpu=332.5us -- a "break-even" where the GPU was
        actually slower. When the two are within CONFIRM_MARGIN we re-time with
        many more repetitions and take the median of medians.
        """
        r = 15 if NP * D < 50_000 else 7
        tc, _ = time_generation("cpu", NP, D, evaluate, reps=r)
        tg, _ = time_generation("cuda", NP, D, evaluate, reps=r)
        if confirm or abs(tg - tc) / max(tc, tg) < CONFIRM_MARGIN:
            cs, gs = [], []
            for _ in range(CONFIRM_ROUNDS):
                a, _ = time_generation("cpu", NP, D, evaluate, reps=r * 2)
                b, _ = time_generation("cuda", NP, D, evaluate, reps=r * 2)
                cs.append(a); gs.append(b)
            tc, tg = float(np.median(cs)), float(np.median(gs))
        return tg < tc, tc, tg

    w, tc, tg = wins(np_floor)
    if w:
        return {"D": D, "np_star": np_floor, "n_star": np_floor * D,
                "bracket": "gpu wins at NP floor", "censored": True,
                "t_cpu_s": tc, "t_gpu_s": tg}
    if not wins(hi)[0]:
        return {"D": D, "np_star": None, "bracket": "no crossover below cap",
                "np_cap": hi}
    lo = np_floor
    while hi / lo > 1.02 and hi - lo > 8:
        mid = max(lo + 1, min(int(math.sqrt(lo * hi)), hi - 1))
        if wins(mid)[0]:
            hi = mid
        else:
            lo = mid
    if hi - lo <= 64:
        for cand in range(lo + 1, hi + 1):
            if wins(cand)[0]:
                hi = cand; break
            lo = cand
    _, tc, tg = wins(hi)
    return {"D": D, "np_star": hi, "n_star": hi * D, "censored": False,
            "t_cpu_s": tc, "t_gpu_s": tg}


# ======================================================================= main
def main():
    print("=" * 78)
    print("  FITNESS-COST SWEEP -- does the break-even scale with evaluation cost?")
    print("=" * 78)
    if not torch.cuda.is_available():
        print("  ERROR: no CUDA device."); sys.exit(1)
    p = torch.cuda.get_device_properties(0)
    env = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu_name": p.name, "gpu_capability": f"{p.major}.{p.minor}",
           "torch_threads": torch.get_num_threads(),
           "platform": platform.platform(), "seed": SEED,
           "numpy": np.__version__}
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    env["cpu_model"] = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    for k in ("cpu_model", "gpu_name", "torch", "cuda"):
        print(f"  {k:<16} {env.get(k)}")

    res = {"environment": env, "config": {
        "grid_np": GRID_NP, "grid_d": GRID_D, "be_dims": BE_DIMS,
        "np_floor": NP_FLOOR, "seed": SEED}, "functions": {}}

    print("\n" + "-" * 78 + "\n  (1) ISOLATED FITNESS COST\n" + "-" * 78)
    print(f"      {'function':<12}{'cpu ns/el':>12}{'gpu ns/el':>12}{'rel(cpu)':>11}")
    costs = {}
    for name, fn in FUNCTIONS:
        cc, _ = fitness_cost("cpu", fn)
        cg, _ = fitness_cost("cuda", fn)
        costs[name] = {"cpu_ns": cc * 1e9, "gpu_ns": cg * 1e9}
        print(f"      {name:<12}{cc*1e9:>12.4f}{cg*1e9:>12.4f}"
              f"{cc/list(costs.values())[0]['cpu_ns']*1e9:>11.2f}x")
    res["fitness_cost"] = costs
    span = (max(v["cpu_ns"] for v in costs.values())
            / min(v["cpu_ns"] for v in costs.values()))
    print(f"\n      cost span: {span:.1f}x")

    print("\n" + "-" * 78 + "\n  (2) COST MODEL PER FUNCTION\n" + "-" * 78)
    print(f"      {'function':<12}{'c_cpu us':>10}{'b_cpu ns':>10}"
          f"{'c_gpu us':>10}{'b_gpu ns':>10}{'db ns':>9}{'R2cpu':>8}")
    fits = {}
    for name, fn in FUNCTIONS:
        fc = fit_full(cost_grid("cpu", fn))
        fg = fit_full(cost_grid("cuda", fn))
        flags = []
        if fc["c_s"] < 0: flags.append("c_cpu<0")
        if fg["c_s"] < 0: flags.append("c_gpu<0")
        if fc["b_s"] < 0: flags.append("b_cpu<0")
        if fg["b_s"] < 0: flags.append("b_gpu<0")
        if fg["c_s"] - fc["c_s"] <= 0: flags.append("dc<=0")
        if fg["r2"] < 0.60: flags.append(f"gpu_r2={fg['r2']:.2f}")
        if fc["r2"] < 0.95: flags.append(f"cpu_r2={fc['r2']:.2f}")
        fits[name] = {"cpu": fc, "gpu": fg,
                      "dc": fg["c_s"] - fc["c_s"],
                      "da": fc["a_s"] - fg["a_s"],
                      "db": fc["b_s"] - fg["b_s"],
                      "flags": flags, "usable": len(flags) == 0}
        if flags:
            print(f"      {'':12}  ^^ UNPHYSICAL/UNRELIABLE: {', '.join(flags)}")
        print(f"      {name:<12}{fc['c_s']*1e6:>10.1f}{fc['b_s']*1e9:>10.3f}"
              f"{fg['c_s']*1e6:>10.1f}{fg['b_s']*1e9:>10.3f}"
              f"{fits[name]['db']*1e9:>9.3f}{fc['r2']:>8.4f}")
    res["fits"] = fits

    print("\n" + "-" * 78 + "\n  (3) OUT-OF-SAMPLE: dc,da FROM SPHERE ONLY\n" + "-" * 78)
    dc0, da0 = fits["sphere"]["dc"], fits["sphere"]["da"]
    print(f"      sphere dc={dc0*1e6:.2f}us  da={da0*1e9:.2f}ns  "
          f"(held fixed for all functions)")
    print(f"\n      {'function':<12}{'D':>5}{'measured':>10}{'predicted':>11}{'err':>9}")
    be = {}
    for name, fn in FUNCTIONS:
        db = fits[name]["db"]
        rows = []
        for D in BE_DIMS:
            m = measure_breakeven(D, fn)
            denom = da0 + db * D
            m["np_star_predicted"] = dc0 / denom if denom > 0 else None
            if m.get("np_star") and not m.get("censored") and m["np_star_predicted"]:
                m["rel_error"] = ((m["np_star_predicted"] - m["np_star"])
                                  / m["np_star"])
                print(f"      {name:<12}{D:>5}{m['np_star']:>10,}"
                      f"{m['np_star_predicted']:>11,.1f}{m['rel_error']:>+9.1%}")
            else:
                print(f"      {name:<12}{D:>5}{'':>10}{'':>11}"
                      f"   {m.get('bracket','')}")
            rows.append(m)
        be[name] = rows
    res["breakeven"] = be

    print("\n" + "-" * 78 + "\n  (4) TREND TEST\n" + "-" * 78)
    xs, ys = [], []
    for name, _ in FUNCTIONS:
        r = [q for q in be[name] if q["D"] == 30 and q.get("np_star")]
        if r:
            xs.append(costs[name]["cpu_ns"]); ys.append(r[0]["np_star"])
            print(f"      {name:<12} fitness {costs[name]['cpu_ns']:>8.3f} ns/el"
                  f"   N*(D=30) = {r[0]['np_star']:>7,}")
    if len(xs) >= 3:
        sl = np.polyfit(np.log(xs), np.log(ys), 1)[0]
        rr = np.corrcoef(np.log(xs), np.log(ys))[0, 1]
        res["trend"] = {"loglog_slope": float(sl), "log_r": float(rr),
                        "fitness_ns": xs, "np_star": ys}
        print(f"\n      log-log slope = {sl:+.3f}   r = {rr:+.4f}")
        print("      negative slope => break-even shrinks as fitness cost rises,")
        print("      which is what the model predicts. Flat => model is wrong.")

    tag = "".join(c if c.isalnum() else "_" for c in env["gpu_name"]).strip("_")
    path = f"fitness_{tag}.json"
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    print("\n" + "=" * 78 + f"\n  WROTE {path}\n" + "=" * 78)


if __name__ == "__main__":
    main()

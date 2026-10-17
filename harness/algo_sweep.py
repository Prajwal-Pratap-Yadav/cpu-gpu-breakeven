#!/usr/bin/env python3
"""
================================================================================
 EXPERIMENT: DOES THE COST MODEL GENERALISE ACROSS POPULATION METHODS?
================================================================================
 THE OBJECTION THIS ANSWERS

 Every result so far used DE/rand/1/bin. A reviewer will ask whether the
 synchronisation account is a property of population-based optimisation, or
 an artefact of one particular kernel.

 THE PREDICTION, STATED BEFORE MEASURING

 The model t = c + a*NP + b*NP*D assumes per-generation work decomposes into
 a fixed cost, a per-individual cost, and an elementwise cost over the
 (NP x D) population tensor.

   (A1) DE, PSO and jDE all update an (NP x D) tensor elementwise and
        synchronise once per generation. The model should hold for all three,
        with similar c (a property of the framework and device, not the
        algorithm) and differing a and b.

   (A2) CMA-ES should BREAK the model. Its rank-mu covariance update is
        O(mu*D^2) and its eigendecomposition is O(D^3); neither term scales
        with NP at all. A model with no D^2 or D^3 term cannot fit it, and
        forcing one produces the negative coefficients we already saw with
        expensive8.

 A1 failing would mean the account is DE-specific and the paper's scope
 collapses to one algorithm. A2 failing -- CMA-ES fitting fine -- would mean
 the model is absorbing structure it should not, and the good fits elsewhere
 are less meaningful than they look. Both outcomes are informative.

 WE ALSO FIT AN EXTENDED FORM

   t = c + a*NP + b*NP*D + e*D^2 + g*D^3

 and report which model each algorithm needs. That converts "CMA-ES breaks it"
 from a limitation into a characterisation: the model applies to methods whose
 per-generation cost is elementwise in the population tensor.

 ADAPTIVE REPETITIONS
   CMA-ES at D=300 runs an eigendecomposition per generation, thousands of
   times slower than a DE step. Fixed repetition counts would make the run
   unbounded, so reps are chosen per measurement to target a fixed time budget.

 Fitness is held at Sphere throughout, to isolate the ALGORITHM effect from
 the fitness-cost effect measured separately.

 KAGGLE: pin torch==2.3.1+cu118, restart, run. Writes algos_<gpu>.json.
================================================================================
"""
import json, math, os, platform, sys, time
from datetime import datetime, timezone

import numpy as np
import torch

SEED       = 20260820
DTYPE      = torch.float32
GRID_NP    = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
GRID_D     = [2, 5, 10, 30, 100, 300]
BE_DIMS    = [10, 30, 100]
NP_FLOOR   = 32
TIME_BUDGET_S   = 0.40      # per measurement; sets reps adaptively
MIN_REPS, MAX_REPS = 5, 31
CONFIRM_MARGIN, CONFIRM_ROUNDS = 0.08, 3
MEM_FRACTION, TEMPS_PER_GEN = 0.20, 12

torch.manual_seed(SEED)
np.random.seed(SEED)


def sphere(z):
    return (z * z).sum(dim=1)


# ============================================================== algorithms
# Each returns a closure stepping one generation on preallocated state, so
# timing measures the generation and not the setup.

def make_de(NP, D, dev):
    x = torch.rand(NP, D, dtype=DTYPE, device=dev)

    def step():
        idx = torch.randint(0, NP, (3, NP), device=dev)
        v = x[idx[0]] + 0.5 * (x[idx[1]] - x[idx[2]])
        mask = torch.rand(NP, D, device=dev) < 0.9
        mask.scatter_(1, torch.randint(0, D, (NP, 1), device=dev), True)
        u = torch.where(mask, v, x)
        better = (sphere(u) < sphere(x)).unsqueeze(1)
        return torch.where(better, u, x)
    return step, [x]


def make_pso(NP, D, dev):
    x = torch.rand(NP, D, dtype=DTYPE, device=dev)
    v = torch.zeros(NP, D, dtype=DTYPE, device=dev)
    pb = x.clone()
    pbf = sphere(x)
    gb = pb[pbf.argmin()].clone()

    def step():
        r1 = torch.rand(NP, D, device=dev)
        r2 = torch.rand(NP, D, device=dev)
        vv = 0.7 * v + 1.5 * r1 * (pb - x) + 1.5 * r2 * (gb - x)
        xx = x + vv
        f = sphere(xx)
        better = f < pbf
        pbb = torch.where(better.unsqueeze(1), xx, pb)
        pbff = torch.where(better, f, pbf)
        return pbb[pbff.argmin()]
    return step, [x, v, pb, pbf, gb]


def make_jde(NP, D, dev):
    x = torch.rand(NP, D, dtype=DTYPE, device=dev)
    F = torch.full((NP,), 0.5, dtype=DTYPE, device=dev)
    CR = torch.full((NP,), 0.9, dtype=DTYPE, device=dev)

    def step():
        r = torch.rand(4, NP, device=dev)
        Fn = torch.where(r[0] < 0.1, 0.1 + 0.9 * r[1], F)
        CRn = torch.where(r[2] < 0.1, r[3], CR)
        idx = torch.randint(0, NP, (3, NP), device=dev)
        v = x[idx[0]] + Fn.unsqueeze(1) * (x[idx[1]] - x[idx[2]])
        mask = torch.rand(NP, D, device=dev) < CRn.unsqueeze(1)
        mask.scatter_(1, torch.randint(0, D, (NP, 1), device=dev), True)
        u = torch.where(mask, v, x)
        better = sphere(u) < sphere(x)
        return torch.where(better.unsqueeze(1), u, x)
    return step, [x, F, CR]


def make_cmaes(NP, D, dev):
    """
    Rank-mu CMA-ES with an eigendecomposition EVERY generation.

    Production CMA-ES amortises the decomposition over ~D/10 generations. We do
    it every generation deliberately: the point is to expose the O(D^3) term,
    not to run a competitive optimiser. This is stated in the paper rather than
    presented as a like-for-like efficiency comparison.
    """
    mu = max(1, NP // 2)
    mean = torch.zeros(D, dtype=DTYPE, device=dev)
    C = torch.eye(D, dtype=DTYPE, device=dev)
    sigma = 0.3
    w = torch.log(torch.tensor(mu + 0.5, device=dev)) - torch.log(
        torch.arange(1, mu + 1, dtype=DTYPE, device=dev))
    w = w / w.sum()

    def step():
        evals, B = torch.linalg.eigh(C)                     # O(D^3)
        d = torch.sqrt(torch.clamp(evals, min=1e-20))
        z = torch.randn(NP, D, dtype=DTYPE, device=dev)
        y = (z * d) @ B.T                                   # O(NP*D^2)
        xs = mean + sigma * y
        f = sphere(xs)
        order = torch.argsort(f)[:mu]
        ysel = y[order]
        newmean = mean + sigma * (w.unsqueeze(1) * ysel).sum(dim=0)
        Cn = (1 - 0.1) * C + 0.1 * (ysel.T * w) @ ysel      # O(mu*D^2)
        return newmean, Cn
    return step, [mean, C, w]


ALGORITHMS = [("de", make_de), ("pso", make_pso),
              ("jde", make_jde), ("cmaes", make_cmaes)]


# =================================================================== timing
def _sum(s):
    a = np.asarray(s, float)
    return {"median_s": float(np.median(a)), "mean_s": float(a.mean()),
            "std_s": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "p05_s": float(np.percentile(a, 5)),
            "p95_s": float(np.percentile(a, 95)), "n": int(a.size)}


def adaptive_time(step, cuda):
    """Choose reps to hit TIME_BUDGET_S, so slow algorithms stay bounded."""
    for _ in range(3):
        step()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    step()
    if cuda:
        torch.cuda.synchronize()
    probe = max(time.perf_counter() - t0, 1e-9)
    reps = int(np.clip(TIME_BUDGET_S / probe, MIN_REPS, MAX_REPS))

    for _ in range(min(5, reps)):
        step()
    if cuda:
        torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        if cuda:
            torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return _sum(out)


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
    return max(8, int(budget / (TEMPS_PER_GEN * D * 4)))


def time_algo(device, maker, NP, D):
    dev = torch.device(device)
    step, state = maker(NP, D, dev)
    st = adaptive_time(step, device == "cuda")
    del step, state
    if device == "cuda":
        torch.cuda.empty_cache()
    return st["median_s"], st


# ==================================================================== fits
def fit_standard(rows):
    NP = np.array([r["NP"] for r in rows], float)
    D = np.array([r["D"] for r in rows], float)
    t = np.array([r["time_s"] for r in rows], float)
    A = np.column_stack([np.ones_like(NP), NP, NP * D])
    return _lsq(A, t, ["c_s", "a_s", "b_s"])


def fit_extended(rows):
    NP = np.array([r["NP"] for r in rows], float)
    D = np.array([r["D"] for r in rows], float)
    t = np.array([r["time_s"] for r in rows], float)
    A = np.column_stack([np.ones_like(NP), NP, NP * D, D ** 2, D ** 3])
    return _lsq(A, t, ["c_s", "a_s", "b_s", "e_s", "g_s"])


def _lsq(A, t, names):
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    pred = A @ coef
    ss_res = float(((t - pred) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    out = {n: float(v) for n, v in zip(names, coef)}
    out["r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out["n_points"] = int(len(t))
    out["max_abs_rel_resid"] = float(np.max(np.abs(t - pred) / t))
    return out


def cost_grid(device, maker):
    rows = []
    for D in GRID_D:
        for NP in GRID_NP:
            if NP > max_np_for(D, device):
                continue
            try:
                t, _ = time_algo(device, maker, NP, D)
            except RuntimeError:
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            rows.append({"NP": NP, "D": D, "time_s": t})
    return rows


# ============================================================== break-even
def measure_breakeven(D, maker):
    hi = min(max_np_for(D, "cuda"), max_np_for(D, "cpu"), 2 ** 20)
    if hi <= NP_FLOOR:
        return {"D": D, "np_star": None, "bracket": "no headroom"}

    def wins(NP):
        tc, _ = time_algo("cpu", maker, NP, D)
        tg, _ = time_algo("cuda", maker, NP, D)
        if abs(tg - tc) / max(tc, tg) < CONFIRM_MARGIN:
            cs, gs = [], []
            for _ in range(CONFIRM_ROUNDS):
                a, _ = time_algo("cpu", maker, NP, D)
                b, _ = time_algo("cuda", maker, NP, D)
                cs.append(a); gs.append(b)
            tc, tg = float(np.median(cs)), float(np.median(gs))
        return tg < tc, tc, tg

    w, tc, tg = wins(NP_FLOOR)
    if w:
        return {"D": D, "np_star": NP_FLOOR, "n_star": NP_FLOOR * D,
                "bracket": "gpu wins at NP floor", "censored": True,
                "t_cpu_s": tc, "t_gpu_s": tg}
    if not wins(hi)[0]:
        return {"D": D, "np_star": None, "bracket": "no crossover below cap",
                "np_cap": hi}
    lo = NP_FLOOR
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
    print("  CROSS-ALGORITHM SWEEP -- does the cost model generalise?")
    print("=" * 78)
    if not torch.cuda.is_available():
        print("  ERROR: no CUDA device."); sys.exit(1)
    p = torch.cuda.get_device_properties(0)
    env = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu_name": p.name, "gpu_capability": f"{p.major}.{p.minor}",
           "torch_threads": torch.get_num_threads(), "numpy": np.__version__,
           "platform": platform.platform(), "seed": SEED}
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
        "np_floor": NP_FLOOR, "time_budget_s": TIME_BUDGET_S,
        "fitness": "sphere", "seed": SEED}, "algorithms": {}}

    print("\n" + "-" * 78)
    print("  (1) COST MODEL -- STANDARD vs EXTENDED (A1 / A2)")
    print("-" * 78)
    print(f"  {'algo':<7}{'dev':<5}{'std R2':>9}{'ext R2':>9}{'gain':>8}"
          f"{'c us':>9}{'a ns':>10}{'b ns':>9}{'D^2 ns':>10}{'D^3 ns':>10}")
    for name, maker in ALGORITHMS:
        res["algorithms"][name] = {}
        for dev in ("cpu", "cuda"):
            rows = cost_grid(dev, maker)
            std, ext = fit_standard(rows), fit_extended(rows)
            res["algorithms"][name][dev] = {
                "grid": rows, "standard": std, "extended": ext,
                "r2_gain": ext["r2"] - std["r2"]}
            print(f"  {name:<7}{dev:<5}{std['r2']:>9.4f}{ext['r2']:>9.4f}"
                  f"{ext['r2']-std['r2']:>+8.4f}"
                  f"{std['c_s']*1e6:>9.1f}{std['a_s']*1e9:>10.1f}"
                  f"{std['b_s']*1e9:>9.3f}"
                  f"{ext['e_s']*1e9:>10.2f}{ext['g_s']*1e9:>10.4f}")

    print("\n  A1 holds if de/pso/jde have high standard R2 and small gain.")
    print("  A2 holds if cmaes has low standard R2 and a large gain.")

    print("\n" + "-" * 78)
    print("  (2) PREDICTED vs MEASURED BREAK-EVEN")
    print("-" * 78)
    print(f"  {'algo':<7}{'D':>5}{'measured':>10}{'predicted':>11}{'err':>9}")
    for name, maker in ALGORITHMS:
        fc = res["algorithms"][name]["cpu"]["standard"]
        fg = res["algorithms"][name]["cuda"]["standard"]
        dc = fg["c_s"] - fc["c_s"]
        da = fc["a_s"] - fg["a_s"]
        db = fc["b_s"] - fg["b_s"]
        res["algorithms"][name]["deltas"] = {"dc": dc, "da": da, "db": db}
        rows = []
        for D in BE_DIMS:
            m = measure_breakeven(D, maker)
            den = da + db * D
            m["np_star_predicted"] = dc / den if (den > 0 and dc > 0) else None
            if m.get("np_star") and m["np_star_predicted"] and not m.get("censored"):
                m["rel_error"] = ((m["np_star_predicted"] - m["np_star"])
                                  / m["np_star"])
                print(f"  {name:<7}{D:>5}{m['np_star']:>10,}"
                      f"{m['np_star_predicted']:>11,.1f}{m['rel_error']:>+9.1%}")
            else:
                print(f"  {name:<7}{D:>5}{m.get('np_star') or '':>10}"
                      f"{'':>11}   {m.get('bracket','no prediction')}")
            rows.append(m)
        res["algorithms"][name]["breakeven"] = rows

    print("\n" + "-" * 78 + "\n  (3) SUMMARY\n" + "-" * 78)
    for name, _ in ALGORITHMS:
        A = res["algorithms"][name]
        errs = [abs(r["rel_error"]) for r in A["breakeven"]
                if r.get("rel_error") is not None]
        med = f"{np.median(errs):.1%}" if errs else "n/a"
        print(f"  {name:<7} cpu std R2={A['cpu']['standard']['r2']:.4f}"
              f"  ext gain={A['cpu']['r2_gain']:+.4f}"
              f"  break-even median err={med}")

    tag = "".join(c if c.isalnum() else "_" for c in env["gpu_name"]).strip("_")
    path = f"algos_{tag}.json"
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    print("\n" + "=" * 78 + f"\n  WROTE {path}\n" + "=" * 78)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
================================================================================
 FINAL EXPERIMENT -- everything remaining, in one run
================================================================================
 Run on T4, then on P100. Writes final_<gpu>.json.

 SECTIONS
   1  environment + preflight
   2  bandwidth hierarchy (CPU and GPU)
   3  cost grid, fitted by NNLS (physical non-negative coefficients)
   4  energy per generation (GPU via nvidia-smi, CPU via RAPL if available)
   5  CUDA graph capture, NP floor lowered to 4 for a QUANTITATIVE break-even
   6  repeatability -- the noise floor bounding every accuracy claim
   7  summary

 ROBUSTNESS
   Every section is wrapped and the JSON is CHECKPOINTED after each one, so a
   crash or a session timeout still leaves usable data. If a section fails the
   run continues and the error is recorded rather than lost.

 WHY NNLS
   c, a and b are physical costs and cannot be negative. Ordinary least squares
   fitted a = -144 ns for PSO and -192 ns for CMA-ES, which says an extra
   individual makes a generation faster. Those unphysical fits produced 44-270%
   break-even errors. NNLS is a RESTRICTION of OLS: it has fewer degrees of
   freedom and lower in-sample R2, yet cut out-of-sample error to 4-15%. A
   constraint that fits training data worse while predicting held-out data
   better is evidence the constraint is correct.
================================================================================
"""
import json, math, os, platform, subprocess, sys, time, traceback
from datetime import datetime, timezone

import numpy as np
import torch

try:
    from scipy.optimize import nnls
    HAVE_NNLS = True
except ImportError:
    HAVE_NNLS = False

SEED        = 20260820
DTYPE       = torch.float32
DTYPE_BYTES = 4
TRIAD_BYTES = 12
GRID_NP     = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
GRID_D      = [2, 5, 10, 30, 100, 300]
BE_DIMS     = [10, 30, 50, 100, 200, 500]
NP_FLOOR_EAGER = 32
NP_FLOOR_GRAPH = 4          # lowered: graphed break-even was censored before
REPS, WARM  = 15, 5
ENERGY_SECONDS = 12.0
REPEAT_CONFIGS = [(10, 3), (30, 3), (100, 3)]
CONFIRM_MARGIN, CONFIRM_ROUNDS = 0.08, 3
MEM_FRACTION, TEMPS_PER_GEN = 0.20, 12
OUT = {}

torch.manual_seed(SEED)
np.random.seed(SEED)


def checkpoint(path):
    try:
        with open(path, "w") as fh:
            json.dump(OUT, fh, indent=2, default=str)
    except Exception as exc:
        print(f"      [checkpoint failed: {exc}]")


def section(name, path, fn, *args, **kw):
    print("\n" + "-" * 78 + f"\n  {name}\n" + "-" * 78)
    try:
        out = fn(*args, **kw)
        checkpoint(path)
        return out
    except Exception as exc:
        print(f"      SECTION FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=2)
        OUT.setdefault("errors", {})[name] = f"{type(exc).__name__}: {exc}"
        checkpoint(path)
        return None


# ================================================================== helpers
def _sum(s):
    a = np.asarray(s, float)
    return {"median_s": float(np.median(a)), "mean_s": float(a.mean()),
            "std_s": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "p05_s": float(np.percentile(a, 5)),
            "p95_s": float(np.percentile(a, 95)),
            "cv": float(a.std(ddof=1) / a.mean()) if a.size > 1 else 0.0,
            "n": int(a.size)}


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


def sphere(z):
    return (z * z).sum(dim=1)


def de_generation(x, evaluate=sphere):
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
    return max(8, int(budget / (TEMPS_PER_GEN * D * DTYPE_BYTES)))


def time_generation(device, NP, D, reps=REPS, warmup=WARM):
    cuda = device == "cuda"
    x = torch.rand(NP, D, dtype=DTYPE, device=torch.device(device))
    st = time_fn(lambda: de_generation(x), cuda, warmup, reps)
    del x
    if cuda:
        torch.cuda.empty_cache()
    return st["median_s"], st


# ============================================================= 1 environment
def do_environment():
    p = torch.cuda.get_device_properties(0)
    env = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "torch": torch.__version__,
           "cuda": torch.version.cuda, "gpu_name": p.name,
           "gpu_capability": f"{p.major}.{p.minor}",
           "gpu_total_mem_bytes": p.total_memory,
           "gpu_sm_count": p.multi_processor_count,
           "torch_threads": torch.get_num_threads(), "cpu_count": os.cpu_count(),
           "seed": SEED, "have_nnls": HAVE_NNLS,
           "torch_arch_list": torch.cuda.get_arch_list()}
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    env["cpu_model"] = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    try:
        env["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,clocks.max.sm,"
             "power.limit,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        pass
    for k in ("cpu_model", "gpu_name", "torch", "cuda", "have_nnls"):
        print(f"      {k:<18} {env.get(k)}")
    if not HAVE_NNLS:
        print("      WARNING: scipy missing -> falling back to clipped OLS")
    OUT["environment"] = env
    return env


# ============================================================== 2 bandwidth
def bandwidth_curve(device, label, kmax=27):
    print(f"\n      [{label}] STREAM triad")
    cuda = device == "cuda"
    rows = []
    for k in range(20, kmax):
        n = 2 ** k
        if not cuda and 3 * n * DTYPE_BYTES > (2 << 30):
            break
        try:
            dev = torch.device(device)
            b = torch.rand(n, dtype=DTYPE, device=dev)
            c = torch.rand(n, dtype=DTYPE, device=dev)
            a = torch.empty_like(b)
            st = time_fn(lambda: [torch.add(b, c, alpha=3.14159, out=a)
                                  for _ in range(10)] and None, cuda, 3, 7)
            bw = TRIAD_BYTES * n / (st["median_s"] / 10)
            rows.append({"n": n, "working_set_bytes": 3 * n * DTYPE_BYTES,
                         "bw_bytes_per_s": bw})
            print(f"        2^{k:<2} {bw/1e9:8.2f} GB/s "
                  f"(working set {3*n*DTYPE_BYTES/1e6:8.1f} MB)")
            del a, b, c
            if cuda:
                torch.cuda.empty_cache()
        except RuntimeError:
            print(f"        2^{k:<2} skipped (OOM)")
            break
    if not rows:
        return {"curve": [], "peak_bytes_per_s": None, "plateau_bytes_per_s": None}
    peak = max(r["bw_bytes_per_s"] for r in rows)
    plateau = rows[-1]["bw_bytes_per_s"]
    print(f"        peak {peak/1e9:.2f}  plateau {plateau/1e9:.2f} GB/s")
    return {"curve": rows, "peak_bytes_per_s": peak,
            "plateau_bytes_per_s": plateau}


def do_bandwidth():
    OUT["bandwidth"] = {"cpu": bandwidth_curve("cpu", "CPU"),
                        "gpu": bandwidth_curve("cuda", "GPU")}
    return OUT["bandwidth"]


# ========================================================== 3 cost grid + fit
def fit_nnls(rows):
    NP = np.array([r["NP"] for r in rows], float)
    D = np.array([r["D"] for r in rows], float)
    t = np.array([r["time_s"] for r in rows], float)
    A = np.column_stack([np.ones_like(NP), NP, NP * D])
    if HAVE_NNLS:
        coef, _ = nnls(A, t)
    else:
        coef = np.clip(np.linalg.lstsq(A, t, rcond=None)[0], 0, None)
    pred = A @ coef
    ss_res = float(((t - pred) ** 2).sum()); ss_tot = float(((t - t.mean()) ** 2).sum())
    ols, *_ = np.linalg.lstsq(A, t, rcond=None)
    pred_o = A @ ols
    r2_o = 1 - float(((t - pred_o) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {"c_s": float(coef[0]), "a_s": float(coef[1]), "b_s": float(coef[2]),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "ols_c_s": float(ols[0]), "ols_a_s": float(ols[1]),
            "ols_b_s": float(ols[2]), "ols_r2": r2_o,
            "n_points": len(rows)}


def do_grid():
    grids = {}
    for dev in ("cpu", "cuda"):
        print(f"\n      [{dev}] cost grid")
        rows = []
        for D in GRID_D:
            line = []
            for NP in GRID_NP:
                if NP > max_np_for(D, dev):
                    continue
                try:
                    t, _ = time_generation(dev, NP, D, reps=11)
                except RuntimeError:
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                    continue
                rows.append({"NP": NP, "D": D, "time_s": t})
                line.append(f"{t*1e6:7.1f}")
            print(f"        D={D:<4} " + " ".join(line))
        grids[dev] = rows
    fits = {d: fit_nnls(grids[d]) for d in grids}
    for d, f in fits.items():
        print(f"\n      {d.upper():<5} NNLS: c={f['c_s']*1e6:8.2f}us "
              f"a={f['a_s']*1e9:8.2f}ns b={f['b_s']*1e9:8.4f}ns  R2={f['r2']:.4f}")
        print(f"      {d.upper():<5} OLS : c={f['ols_c_s']*1e6:8.2f}us "
              f"a={f['ols_a_s']*1e9:8.2f}ns b={f['ols_b_s']*1e9:8.4f}ns  "
              f"R2={f['ols_r2']:.4f}")
    OUT["grid"] = grids
    OUT["fit"] = fits
    dc = fits["cuda"]["c_s"] - fits["cpu"]["c_s"]
    da = fits["cpu"]["a_s"] - fits["cuda"]["a_s"]
    db = fits["cpu"]["b_s"] - fits["cuda"]["b_s"]
    OUT["deltas"] = {"dc": dc, "da": da, "db": db,
                     "asymptotic_n_star": dc / db if db > 0 else None}
    print(f"\n      dc={dc*1e6:.2f}us  da={da*1e9:.2f}ns  db={db*1e9:.4f}ns")
    if db > 0:
        print(f"      asymptotic N* = {dc/db:,.0f}")
    return fits


# ================================================================= 4 energy
def read_rapl():
    for base in ("/sys/class/powercap/intel-rapl:0/energy_uj",
                 "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"):
        try:
            with open(base) as fh:
                return int(fh.read().strip()), base
        except Exception:
            continue
    return None, None


def gpu_power_w():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=power.draw",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        return float(o.stdout.strip().split("\n")[0])
    except Exception:
        return None


def do_energy():
    if gpu_power_w() is None:
        print("      nvidia-smi power.draw unavailable on this device.")
        print("      (T4 reports it; some virtualised GPUs do not.)")
        OUT["energy"] = {"available": False}
        return None

    print("      sampling idle GPU power for 4 s...")
    idle = [gpu_power_w() for _ in range(8) if not time.sleep(0.5)]
    idle = [p for p in idle if p]
    idle_w = float(np.median(idle)) if idle else 0.0
    print(f"      idle {idle_w:.2f} W")

    rapl0, rapl_path = read_rapl()
    print(f"      CPU RAPL: {'available at ' + rapl_path if rapl0 else 'NOT available'}")

    results = []
    for NP, D in [(256, 30), (4096, 30), (4096, 300)]:
        row = {"NP": NP, "D": D}
        for dev in ("cpu", "cuda"):
            if NP > max_np_for(D, dev):
                continue
            x = torch.rand(NP, D, dtype=DTYPE, device=torch.device(dev))
            for _ in range(5):
                de_generation(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            r0, _ = read_rapl()
            samples, n_gen = [], 0
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < ENERGY_SECONDS:
                for _ in range(20):
                    de_generation(x)
                    n_gen += 1
                if dev == "cuda":
                    torch.cuda.synchronize()
                    p = gpu_power_w()
                    if p:
                        samples.append(p)
            elapsed = time.perf_counter() - t0
            r1, _ = read_rapl()
            e = {"n_gen": n_gen, "elapsed_s": elapsed,
                 "gen_per_s": n_gen / elapsed}
            if dev == "cuda" and samples:
                mean_w = float(np.median(samples))
                e.update({"mean_power_w": mean_w, "idle_w": idle_w,
                          "total_j_per_gen": mean_w * elapsed / n_gen,
                          "dynamic_j_per_gen": (mean_w - idle_w) * elapsed / n_gen})
            if r0 and r1 and r1 > r0:
                e["cpu_pkg_j_per_gen"] = (r1 - r0) / 1e6 / n_gen
            row[dev] = e
            del x
            if dev == "cuda":
                torch.cuda.empty_cache()
        results.append(row)
        c, g = row.get("cpu", {}), row.get("cuda", {})
        print(f"      NP={NP:<5} D={D:<4} "
              f"cpu {c.get('gen_per_s', 0):8.1f} gen/s"
              + (f" {c['cpu_pkg_j_per_gen']*1000:8.2f} mJ/gen"
                 if "cpu_pkg_j_per_gen" in c else "")
              + f" | gpu {g.get('gen_per_s', 0):8.1f} gen/s"
              + (f" {g['total_j_per_gen']*1000:8.2f} mJ/gen"
                 if "total_j_per_gen" in g else ""))
    OUT["energy"] = {"available": True, "idle_w": idle_w,
                     "rapl": bool(rapl0), "rows": results}
    return results


# ========================================================== 5 graph capture
def build_graph(NP, D, init=None):
    dev = torch.device("cuda")
    sx = torch.empty(NP, D, dtype=DTYPE, device=dev)
    sx.copy_(init if init is not None
             else torch.rand(NP, D, dtype=DTYPE, device=dev))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            sx.copy_(de_generation(sx))
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        sx.copy_(de_generation(sx))
    if init is not None:
        sx.copy_(init)
    return g, sx


def verify_graph(NP=64, D=10, n_gen=40, n_runs=30):
    from scipy import stats
    init0 = torch.rand(NP, D, dtype=DTYPE, device="cuda")
    g, sx = build_graph(NP, D, init=init0)
    g.replay(); torch.cuda.synchronize(); a1 = sx.clone()
    g.replay(); torch.cuda.synchronize(); a2 = sx.clone()
    advanced = bool(not torch.allclose(a1, a2))
    del g, sx; torch.cuda.empty_cache()

    eager, graphed = [], []
    for r in range(n_runs):
        torch.manual_seed(1000 + r)
        init = torch.rand(NP, D, dtype=DTYPE, device="cuda")
        x = init.clone()
        for _ in range(n_gen):
            x = de_generation(x)
        eager.append(float(sphere(x).min().item()))
        gg, sxx = build_graph(NP, D, init=init)
        for _ in range(n_gen):
            gg.replay()
        torch.cuda.synchronize()
        graphed.append(float(sphere(sxx).min().item()))
        del gg, sxx; torch.cuda.empty_cache()
    ks, p = stats.ks_2samp(eager, graphed)
    ok = advanced and p > 0.05
    print(f"      replays advance: {advanced}")
    print(f"      eager median {np.median(eager):.6e} | "
          f"graphed median {np.median(graphed):.6e}")
    print(f"      KS={ks:.4f} p={p:.4f}  VERDICT: {'PASS' if ok else 'FAIL'}")
    return {"pass": bool(ok), "replays_advance": advanced,
            "ks": float(ks), "p": float(p),
            "eager": eager, "graphed": graphed}


def graph_time(NP, D, reps):
    g, sx = build_graph(NP, D)
    st = time_fn(lambda: g.replay(), True, 5, reps)
    del g, sx; torch.cuda.empty_cache()
    return st["median_s"]


def breakeven(D, gpu_time, floor):
    hi = min(max_np_for(D, "cuda"), max_np_for(D, "cpu"), 2 ** 20)
    if hi <= floor:
        return {"D": D, "np_star": None, "bracket": "no headroom"}

    def wins(NP):
        r = 15 if NP * D < 50_000 else 7
        tc, _ = time_generation("cpu", NP, D, reps=r)
        tg = gpu_time(NP, D, r)
        if abs(tg - tc) / max(tc, tg) < CONFIRM_MARGIN:
            cs, gs = [], []
            for _ in range(CONFIRM_ROUNDS):
                a, _ = time_generation("cpu", NP, D, reps=r * 2)
                cs.append(a); gs.append(gpu_time(NP, D, r * 2))
            tc, tg = float(np.median(cs)), float(np.median(gs))
        return tg < tc, tc, tg

    w, tc, tg = wins(floor)
    if w:
        return {"D": D, "np_star": floor, "n_star": floor * D, "censored": True,
                "bracket": f"gpu wins at NP={floor}", "t_cpu_s": tc, "t_gpu_s": tg}
    if not wins(hi)[0]:
        return {"D": D, "np_star": None, "bracket": "no crossover below cap",
                "np_cap": hi}
    lo = floor
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


def do_graph():
    v = verify_graph()
    OUT["graph_verification"] = v
    if not v["pass"]:
        print("      timings suppressed: correctness gate failed")
        return None
    c_graph = graph_time(8, 1, 101)
    c_gpu = time_generation("cuda", 8, 1, reps=101)[0]
    c_cpu = time_generation("cpu", 8, 1, reps=101)[0]
    print(f"\n      c_gpu eager   {c_gpu*1e6:8.2f} us")
    print(f"      c_gpu graphed {c_graph*1e6:8.2f} us  ({c_gpu/c_graph:.2f}x)")
    print(f"      c_cpu         {c_cpu*1e6:8.2f} us")
    OUT["graph_costs"] = {"c_gpu_eager_s": c_gpu, "c_gpu_graph_s": c_graph,
                          "c_cpu_s": c_cpu, "reduction": c_gpu / c_graph}
    print(f"\n      graphed break-even (NP floor {NP_FLOOR_GRAPH}):")
    rows = []
    for D in BE_DIMS:
        m = breakeven(D, graph_time, NP_FLOOR_GRAPH)
        rows.append(m)
        if m.get("np_star"):
            print(f"        D={D:<4} NP*={m['np_star']:>7,}  N*={m['n_star']:>9,}"
                  + ("   [censored]" if m.get("censored") else ""))
        else:
            print(f"        D={D:<4} {m['bracket']}")
    OUT["breakeven_graphed"] = rows
    return rows


def do_eager_breakeven():
    d = OUT.get("deltas", {})
    dc, da, db = d.get("dc"), d.get("da"), d.get("db")
    rows = []
    for D in BE_DIMS:
        m = breakeven(D, lambda NP, DD, r: time_generation("cuda", NP, DD, reps=r)[0],
                      NP_FLOOR_EAGER)
        den = (da + db * D) if (da is not None and db is not None) else None
        m["np_star_predicted"] = (dc / den if den and den > 0 and dc and dc > 0
                                  else None)
        if m.get("np_star") and m["np_star_predicted"] and not m.get("censored"):
            m["rel_error"] = ((m["np_star_predicted"] - m["np_star"])
                              / m["np_star"])
            print(f"      D={D:<4} meas={m['np_star']:>7,} "
                  f"pred={m['np_star_predicted']:>9,.1f} "
                  f"err={m['rel_error']:>+7.1%}  N*={m['n_star']:>9,}")
        else:
            print(f"      D={D:<4} meas={m.get('np_star') or '-':>7} "
                  f"  {m.get('bracket','')}")
        rows.append(m)
    OUT["breakeven_eager"] = rows
    return rows


# ========================================================== 6 repeatability
def do_repeatability():
    out = []
    for D, n in REPEAT_CONFIGS:
        vals = []
        for i in range(n):
            m = breakeven(D, lambda NP, DD, r: time_generation("cuda", NP, DD,
                                                               reps=r)[0],
                          NP_FLOOR_EAGER)
            if m.get("np_star"):
                vals.append(m["np_star"])
        if len(vals) >= 2:
            a = np.array(vals, float)
            row = {"D": D, "values": vals, "mean": float(a.mean()),
                   "cv": float(a.std(ddof=1) / a.mean()),
                   "spread": float(a.max() / a.min())}
            print(f"      D={D:<4} {vals}  mean {a.mean():,.0f}  "
                  f"CV {row['cv']:.1%}  spread {row['spread']:.2f}x")
        else:
            row = {"D": D, "values": vals, "note": "insufficient"}
        out.append(row)
    OUT["repeatability"] = out
    return out


# ======================================================================= main
def main():
    print("=" * 78)
    print("  FINAL EXPERIMENT -- bandwidth, cost model, energy, graphs, repeatability")
    print("=" * 78)
    if not torch.cuda.is_available():
        print("  ERROR: no CUDA device. Set Accelerator -> GPU.")
        sys.exit(1)
    try:
        t = torch.ones(1024, device="cuda"); _ = (t * 2).sum().item()
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"  PREFLIGHT FAILED: {exc}")
        print(f"  torch archs: {torch.cuda.get_arch_list()}")
        print("  Pin torch==2.3.1+cu118 and RESTART the session.")
        sys.exit(1)

    env = do_environment()
    tag = "".join(c if c.isalnum() else "_" for c in env["gpu_name"]).strip("_")
    path = f"final_{tag}.json"
    checkpoint(path)

    section("(2) BANDWIDTH HIERARCHY", path, do_bandwidth)
    section("(3) COST GRID + NNLS FIT", path, do_grid)
    section("(4) ENERGY PER GENERATION", path, do_energy)
    section("(5a) EAGER BREAK-EVEN", path, do_eager_breakeven)
    section("(5b) CUDA GRAPH CAPTURE", path, do_graph)
    section("(6) REPEATABILITY", path, do_repeatability)

    print("\n" + "=" * 78)
    print(f"  WROTE {path}")
    if "errors" in OUT:
        print(f"  SECTIONS WITH ERRORS: {list(OUT['errors'])}")
        print("  (partial data was still saved -- send the file anyway)")
    print("=" * 78)


if __name__ == "__main__":
    main()

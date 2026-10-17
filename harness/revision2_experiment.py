#!/usr/bin/env python3
"""
================================================================================
 REVISION EXPERIMENT, PART 2 (paper 2095) -- closes two gaps left by part 1
================================================================================
 S8  Break-even under CUDA graph capture: graph replay vs host over NP = 1..4096
     (scan + bisection) at D = 10, 30, 100, 500, the eager break-even at the same
     D in the same session, and fixed-cost probes. Correctness gate re-run.  [R2.3]
 S9  Same-session calibration: fixed-cost probes at the start and end, a cost
     grid measured with host and GPU interleaved point by point, and two
     independent break-even measurements at D = 10, 30, 50, 100, 200.
     Tests whether same-session calibration removes the prediction drift. [R2.4]
 Writes revision2_<gpu>.json and revision2_<gpu>.log (saved after each section).
 Local logic test without a GPU:  python3 revision2_experiment.py --dry-run
================================================================================
"""
import glob, json, math, os, platform, random, re, shutil, subprocess, sys, tempfile, time, traceback
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import torch

try:
    from scipy.optimize import nnls
    from scipy import stats as sstats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

DRY = "--dry-run" in sys.argv
ACC = "cpu" if DRY else "cuda"           # accelerator device string
IS_CUDA = ACC == "cuda"
SEED = 20260820
DTYPE = torch.float32
GRID_NP = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
GRID_D = [2, 5, 10, 30, 100, 300]
BE_DIMS = [10, 30, 50, 100, 200]
SIG_DIMS = [10, 30, 50, 100, 200, 500]
HIGH_REPS = 60
BE_REPEATS = 3
ROUNDS = 4
MEM_FRACTION, TEMPS = 0.20, 24
OUT = {}
if DRY:   # keep the local logic test short
    GRID_NP = [16, 64, 256, 1024]
    GRID_D = [2, 10, 100]
    BE_DIMS = [10, 100]
    SIG_DIMS = [10, 100]
    HIGH_REPS = 8
    BE_REPEATS = 2

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# =============================================================== utilities
def sync():
    if IS_CUDA:
        torch.cuda.synchronize()


def save(path):
    with open(path, "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)


ONLY = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)


def section(title, path, fn):
    if ONLY and title.split()[0] not in ONLY:
        return
    print("\n" + "-" * 78 + f"\n  {title}\n" + "-" * 78, flush=True)
    t0 = time.time()
    s0 = cpu_times()
    try:
        fn()
    except Exception as exc:
        print(f"  SECTION FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        OUT.setdefault("errors", {})[title] = f"{type(exc).__name__}: {exc}"
    OUT.setdefault("section_steal", {})[title] = steal_fraction(s0, cpu_times())
    OUT.setdefault("section_seconds", {})[title] = time.time() - t0
    save(path)
    print(f"  [{time.time()-t0:.0f} s, host steal {OUT['section_steal'][title]*100:.2f}%]")


def cpu_times():
    try:
        with open("/proc/stat") as fh:
            return [int(v) for v in fh.readline().split()[1:]]
    except Exception:
        return None


def steal_fraction(a, b):
    """Share of host CPU time taken by the hypervisor (co-tenancy) between a and b."""
    if not a or not b or len(a) < 8:
        return float("nan")
    d = [y - x for x, y in zip(a, b)]
    total = sum(d[:8])
    return d[7] / total if total > 0 else float("nan")


def summarize(s):
    a = np.asarray(s, float)
    q25, q75 = np.percentile(a, [25, 75])
    return {"median_s": float(np.median(a)), "mean_s": float(a.mean()),
            "p05_s": float(np.percentile(a, 5)), "p95_s": float(np.percentile(a, 95)),
            "iqr_over_median": float((q75 - q25) / np.median(a)), "n": int(a.size)}


def time_block(fn, cuda, reps):
    out = []
    for _ in range(reps):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return out


def time_fn(fn, cuda, reps=15, warmup=5):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return summarize(time_block(fn, cuda, reps))


def interleaved(fn_cpu, fn_acc, reps, rounds=ROUNDS, warmup=5):
    """Alternate host and accelerator blocks in random order, so slow drifts and
    co-tenant bursts hit both devices rather than biasing one of them."""
    for _ in range(warmup):
        fn_cpu(); fn_acc()
    sync()
    cs, gs = [], []
    per = max(1, reps // rounds)
    for _ in range(rounds):
        order = [("cpu", fn_cpu), ("acc", fn_acc)]
        random.shuffle(order)
        for name, f in order:
            (cs if name == "cpu" else gs).extend(time_block(f, name == "acc" and IS_CUDA, per))
    return summarize(cs), summarize(gs)


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
    return max(8, int(budget / (TEMPS * D * 4)))


# ============================================================ objectives
def sphere(z):
    return (z * z).sum(dim=1)


def rastrigin(z):
    return 10 * z.shape[1] + (z * z - 10 * torch.cos(2 * math.pi * z)).sum(dim=1)


def ackley(z):
    d = z.shape[1]
    return (-20 * torch.exp(-0.2 * torch.sqrt((z * z).sum(dim=1) / d))
            - torch.exp(torch.cos(2 * math.pi * z).sum(dim=1) / d) + 20 + math.e)


def expensive8(z):
    y = z
    for _ in range(8):
        y = torch.sin(y) * torch.cos(y) + torch.sqrt(torch.abs(y) + 1e-9)
    return (y * y).sum(dim=1)


OBJECTIVES = [("sphere", sphere), ("rastrigin", rastrigin),
              ("ackley", ackley), ("expensive8", expensive8)]


# ============================================================ algorithms
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


def make_de(NP, D, dev):
    x = torch.rand(NP, D, dtype=DTYPE, device=dev)
    return lambda: de_generation(x)


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
    return step


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
    return step


ALGOS = {"de": make_de, "jde": make_jde, "pso": make_pso}


# ================================================================== fits
def fit_nnls(rows):
    NP = np.array([r["NP"] for r in rows], float)
    D = np.array([r["D"] for r in rows], float)
    t = np.array([r["time_s"] for r in rows], float)
    A = np.column_stack([np.ones_like(NP), NP, NP * D])
    if HAVE_SCIPY:
        coef, _ = nnls(A, t)
    else:
        coef = np.clip(np.linalg.lstsq(A, t, rcond=None)[0], 0, None)
    pred = A @ coef
    ss_res = float(((t - pred) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return {"c_s": float(coef[0]), "a_s": float(coef[1]), "b_s": float(coef[2]),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), "n": len(rows)}


def grid(maker, device, reps):
    rows = []
    cuda = device == "cuda"
    for D in GRID_D:
        for NP in GRID_NP:
            if NP > max_np_for(D, device):
                continue
            try:
                f = maker(NP, D, torch.device(device))
                st = time_fn(f, cuda, reps=reps, warmup=3)
            except RuntimeError:
                if cuda:
                    torch.cuda.empty_cache()
                continue
            rows.append({"NP": NP, "D": D, "time_s": st["median_s"],
                         "iqr_over_median": st["iqr_over_median"]})
            del f
            if cuda:
                torch.cuda.empty_cache()
    return rows


def breakeven(D, maker, reps, floor=32, confirm=0.08):
    hi = min(max_np_for(D, ACC), max_np_for(D, "cpu"), 2 ** 18)
    disp = []

    def wins(NP):
        fc = maker(NP, D, torch.device("cpu"))
        fa = maker(NP, D, torch.device(ACC))
        sc, sa = interleaved(fc, fa, reps)
        tc, ta = sc["median_s"], sa["median_s"]
        disp.append(max(sc["iqr_over_median"], sa["iqr_over_median"]))
        if abs(ta - tc) / max(ta, tc) < confirm:
            sc2, sa2 = interleaved(fc, fa, reps * 2)
            tc, ta = sc2["median_s"], sa2["median_s"]
        return ta < tc, tc, ta

    w, tc, ta = wins(floor)
    if w:
        return {"D": D, "np_star": floor, "censored": True, "t_cpu_s": tc, "t_acc_s": ta}
    if not wins(hi)[0]:
        return {"D": D, "np_star": None, "bracket": "no crossover below cap", "np_cap": hi}
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
                hi = cand
                break
            lo = cand
    _, tc, ta = wins(hi)
    return {"D": D, "np_star": hi, "censored": False, "t_cpu_s": tc, "t_acc_s": ta,
            "median_probe_dispersion": float(np.median(disp)) if disp else None}


def predict(fc, fa, D):
    dc = fa["c_s"] - fc["c_s"]
    da = fc["a_s"] - fa["a_s"]
    db = fc["b_s"] - fa["b_s"]
    den = da + db * D
    return dc / den if (dc > 0 and den > 0) else None


# ================================================================ STREAM
def stream_bw(device, sizes):
    cuda = device == "cuda"
    rows = []
    for k in sizes:
        n = 2 ** k
        try:
            b = torch.rand(n, device=device)
            c = torch.rand(n, device=device)
            a = torch.empty_like(b)

            def run():
                for _ in range(10):
                    torch.add(b, c, alpha=3.14159, out=a)
            st = time_fn(run, cuda, reps=7, warmup=3)
            rows.append({"n": n, "bw_bytes_per_s": 12 * n / (st["median_s"] / 10)})
            del a, b, c
            if cuda:
                torch.cuda.empty_cache()
        except RuntimeError:
            break
    return {"curve": rows, "plateau_bytes_per_s": rows[-1]["bw_bytes_per_s"] if rows else None}



def build_graph(NP, D, init=None):
    sx = torch.empty(NP, D, dtype=DTYPE, device="cuda")
    sx.copy_(init if init is not None else torch.rand(NP, D, device="cuda"))
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




# ================================================================== probes
def probes(reps=101):
    """Fixed-cost probes at NP = 8, D = 1, host and GPU interleaved."""
    xc = torch.rand(8, 1)
    xa = torch.rand(8, 1, device=ACC)
    sc, sa = interleaved(lambda: de_generation(xc), lambda: de_generation(xa), reps, rounds=4)
    out = {"c_cpu_s": sc["median_s"], "c_gpu_eager_s": sa["median_s"]}
    if IS_CUDA:
        g, sx = build_graph(8, 1)
        sg = time_fn(lambda: g.replay(), True, reps=reps, warmup=10)
        out["c_gpu_graph_s"] = sg["median_s"]
        del g, sx
        torch.cuda.empty_cache()
    return out


def ks_gate():
    eager, graphed = [], []
    for r in range(30):
        torch.manual_seed(1000 + r)
        init = torch.rand(64, 10, device="cuda")
        x = init.clone()
        for _ in range(40):
            x = de_generation(x)
        eager.append(float(sphere(x).min()))
        g, sx = build_graph(64, 10, init=init)
        for _ in range(40):
            g.replay()
        torch.cuda.synchronize()
        graphed.append(float(sphere(sx).min()))
        del g, sx
    ks = sstats.ks_2samp(eager, graphed)
    return {"ks": float(ks.statistic), "p": float(ks.pvalue)}


# ============================================================ S8 graphed BE
def graph_vs_host(NP, D, reps=60):
    g, sx = build_graph(NP, D)
    xc = torch.rand(NP, D)
    sc, sa = interleaved(lambda: de_generation(xc), lambda: g.replay(), reps, rounds=4)
    tc, tg = sc["median_s"], sa["median_s"]
    if abs(tg - tc) / max(tg, tc) < 0.05:          # close call: re-time with doubled repetitions
        sc, sa = interleaved(lambda: de_generation(xc), lambda: g.replay(), reps * 2, rounds=4)
        tc, tg = sc["median_s"], sa["median_s"]
    del g, sx
    torch.cuda.empty_cache()
    return tg < tc, tc, tg


def graphed_breakeven(D):
    scan = []
    cap = min(4096, max_np_for(D, "cuda"), max_np_for(D, "cpu"))
    nps = [n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096) if n <= cap]
    first = None
    for i, NP in enumerate(nps):
        w, tc, tg = graph_vs_host(NP, D)
        scan.append({"NP": NP, "cpu_us": tc * 1e6, "graph_us": tg * 1e6, "gpu_faster": w})
        print(f"    D={D:<4} NP={NP:<5} host {tc*1e6:8.1f} us  graph {tg*1e6:8.1f} us  "
              f"{'GPU faster' if w else 'host faster'}")
        # stop once the GPU wins at two consecutive scan points
        if w and i > 0 and scan[-2]["gpu_faster"]:
            first = nps[i - 1]
            break
        if w and i == 0:
            continue
    res = {"D": D, "scan": scan}
    if first is None:
        if scan and all(s["gpu_faster"] for s in scan[:2]):
            res.update(np_star=1, censored=True, note="GPU faster from NP = 1")
        else:
            res.update(np_star=None, note="no stable crossover up to NP cap")
        return res
    if first == 1 or (len(scan) >= 2 and scan[0]["gpu_faster"] and scan[1]["gpu_faster"]):
        res.update(np_star=1, censored=True, note="GPU faster from NP = 1")
        return res
    lo = max(s["NP"] for s in scan if s["NP"] < first and not s["gpu_faster"])
    hi = first
    while hi - lo > 1:
        mid = (lo + hi) // 2 if hi - lo <= 16 else int(math.sqrt(lo * hi))
        mid = min(max(mid, lo + 1), hi - 1)
        w, _, _ = graph_vs_host(mid, D)
        if w:
            hi = mid
        else:
            lo = mid
    res.update(np_star=hi, censored=False)
    return res


def s8_graphed():
    if not IS_CUDA:
        OUT["s8_graphed"] = {"skipped": "dry run"}
        return
    gate = ks_gate()
    print(f"  correctness gate: KS p = {gate['p']:.3f}")
    pr = probes()
    print(f"  probes: host {pr['c_cpu_s']*1e6:.1f} us, GPU eager {pr['c_gpu_eager_s']*1e6:.1f} us, "
          f"GPU graph {pr['c_gpu_graph_s']*1e6:.1f} us")
    rows = []
    for D in (10, 30, 100, 500):
        gb = graphed_breakeven(D)
        eb = breakeven(D, make_de, 60)
        gb["eager"] = eb
        rows.append(gb)
        print(f"  D={D:<4} graphed break-even {gb.get('np_star')} ({gb.get('note', '')})   "
              f"eager break-even {eb.get('np_star')}")
    OUT["s8_graphed"] = {"gate": gate, "probes": pr, "rows": rows}


# ========================================================= S9 calibration
def grid_interleaved(reps=30):
    rc, ra = [], []
    for D in GRID_D:
        for NP in GRID_NP:
            if NP > min(max_np_for(D, ACC), max_np_for(D, "cpu")):
                continue
            fc = make_de(NP, D, torch.device("cpu"))
            fa = make_de(NP, D, torch.device(ACC))
            sc, sa = interleaved(fc, fa, reps, rounds=3, warmup=3)
            rc.append({"NP": NP, "D": D, "time_s": sc["median_s"]})
            ra.append({"NP": NP, "D": D, "time_s": sa["median_s"]})
            del fc, fa
    if IS_CUDA:
        torch.cuda.empty_cache()
    return rc, ra


def s9_calibration():
    s0 = cpu_times()
    p_start = probes()
    print(f"  probes (start): host {p_start['c_cpu_s']*1e6:.1f} us, GPU {p_start['c_gpu_eager_s']*1e6:.1f} us")
    gc, ga = grid_interleaved()
    fc, fa = fit_nnls(gc), fit_nnls(ga)
    print(f"  interleaved grid fit: host c={fc['c_s']*1e6:.1f} us a={fc['a_s']*1e9:.1f} ns b={fc['b_s']*1e9:.3f} ns "
          f"(R2 {fc['r2']:.4f}) | GPU c={fa['c_s']*1e6:.1f} us (R2 {fa['r2']:.4f})")
    rows = []
    for D in BE_DIMS:
        for k in range(2):
            m = breakeven(D, make_de, HIGH_REPS)
            m["repeat"] = k
            rows.append(m)
        print(f"    D={D:<4} measured {[r.get('np_star') for r in rows if r['D'] == D]}  "
              f"predicted (fit) {predict(fc, fa, D) or 0:.0f}")
    p_end = probes()
    print(f"  probes (end):   host {p_end['c_cpu_s']*1e6:.1f} us, GPU {p_end['c_gpu_eager_s']*1e6:.1f} us")
    OUT["s9_calibration"] = {"probes_start": p_start, "probes_end": p_end, "grid_cpu": gc, "grid_acc": ga,
                             "fit_cpu": fc, "fit_acc": fa, "breakeven": rows,
                             "steal_fraction": steal_fraction(s0, cpu_times())}


# ================================================================== main
class Tee:
    """Copy everything printed into a log file as well as the screen."""
    def __init__(self, path):
        self.f = open(path, "w")
        self.o = sys.stdout

    def write(self, t):
        self.o.write(t)
        self.f.write(t)
        self.f.flush()

    def flush(self):
        self.o.flush()
        self.f.flush()



def main():
    print("=" * 78 + "\n  REVISION EXPERIMENT, PART 2 (paper 2095)" + ("  [DRY RUN ON CPU]" if DRY else "") + "\n" + "=" * 78)
    if not DRY:
        if not torch.cuda.is_available():
            print("  ERROR: no GPU. Settings -> Accelerator -> GPU T4 x2, then run again.")
            sys.exit(1)
    env = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "numpy": np.__version__, "python": sys.version.split()[0],
           "platform": platform.platform(), "threads": torch.get_num_threads(), "seed": SEED,
           "scipy": HAVE_SCIPY, "dry_run": DRY}
    try:
        with open("/proc/cpuinfo") as fh:
            env["cpu_model"] = next(l.split(":", 1)[1].strip() for l in fh if l.startswith("model name"))
    except Exception:
        pass
    if IS_CUDA:
        p = torch.cuda.get_device_properties(0)
        env.update({"gpu_name": p.name, "gpu_capability": f"{p.major}.{p.minor}", "gpu_mem_bytes": p.total_memory})
    OUT["environment"] = env
    tag = "".join(c if c.isalnum() else "_" for c in env.get("gpu_name", "cpu_dry_run")).strip("_")
    path = f"revision2_{tag}.json"
    log = Tee(f"revision2_{tag}.log")
    log.f.write(f"(log started; environment {json.dumps(env, default=str)})\n")
    sys.stdout = log
    for k in ("cpu_model", "gpu_name", "torch", "cuda"):
        print(f"  {k:<10} {env.get(k)}")
    save(path)
    section("S8 BREAK-EVEN UNDER CUDA GRAPH CAPTURE", path, s8_graphed)
    section("S9 SAME-SESSION CALIBRATION", path, s9_calibration)
    print("\n" + "=" * 78 + f"\n  WROTE {path}")
    if "errors" in OUT:
        print(f"  sections with errors: {list(OUT['errors'])}  (partial data saved; send the file)")
    print("=" * 78)


if __name__ == "__main__":
    main()

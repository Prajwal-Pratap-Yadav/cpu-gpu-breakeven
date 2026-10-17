#!/usr/bin/env python3
"""
================================================================================
 REVISION EXPERIMENT -- measurements requested by the reviewers (paper 2095)
================================================================================
 Run once on Kaggle "GPU T4 x2" and once on "GPU P100", with PyTorch 2.3.1+cu118
 pinned (same build as every earlier run). Writes revision_<gpu>.json and
 checkpoints it after every section, so an interrupted session still yields
 usable data.

 SECTIONS
   S1  Kernel trace of one generation (torch.profiler / CUPTI activity API):
       CUDA kernels per generation, GPU-busy time vs wall time, per-kernel
       durations. Also kernel counts per objective function.        [R1.2, R2.5]
   S2  Hardware DRAM counters (Nsight Compute / nvprof / CUPTI metrics).
       Attempted; the outcome (values or the exact refusal) is recorded. [R1.2]
   S3  Timing-derived DRAM traffic per element at large N, where every tensor
       exceeds L2, plus the achieved bandwidth measured in the same session. [R1.2]
   S4  CUDA-graph replay vs host at NP = 1, 2, 3, 4, 8, 16, interleaved.  [R2.3]
   S5  High-repetition (60 per point), interleaved re-measurement of the cost
       grid and the break-even, with hypervisor steal time recorded.   [R2.4]
   S7  Objective-cost sweep (4 objectives): grids, NNLS, break-even.    [R2.5]
   S6  Six-dimension break-even signature for DE, jDE and PSO.          [R1.4]

 Local logic test without a GPU:  python3 revision_experiment.py --dry-run
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


# =============================================================== S1 trace
def chrome_kernels(prof, n_gen):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    prof.export_chrome_trace(path)
    with open(path) as fh:
        ev = json.load(fh)["traceEvents"]
    os.remove(path)
    kern = [e for e in ev if e.get("ph") == "X"
            and str(e.get("cat", "")).lower() in ("kernel", "gpu_memcpy", "gpu_memset")]
    rt = [e for e in ev if e.get("ph") == "X" and str(e.get("cat", "")).lower() == "cuda_runtime"]
    launches = [e for e in rt if "Launch" in e.get("name", "")]
    syncs = [e for e in rt if "Synchronize" in e.get("name", "")]
    names = Counter(e.get("name", "")[:60] for e in kern)
    kern.sort(key=lambda e: float(e.get("ts", 0)))
    per = max(1, round(len(kern) / n_gen))
    last = [(e.get("name", "")[:90], float(e.get("dur", 0))) for e in kern[-per:]]
    return {
        "kernels_per_gen": len(kern) / n_gen,
        "launch_calls_per_gen": len(launches) / n_gen,
        "sync_calls_per_gen": len(syncs) / n_gen,
        "busy_us_per_gen": sum(float(e.get("dur", 0)) for e in kern) / n_gen,
        "last_generation_kernels": last,
        "top_kernels": names.most_common(25),
    }


def profile_gen(fn, n_gen=20):
    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if IS_CUDA else [])
    for _ in range(10):
        fn()
    sync()
    with profile(activities=acts) as prof:
        for _ in range(n_gen):
            fn()
            sync()
    return chrome_kernels(prof, n_gen)


def s1_trace():
    bw = stream_bw(ACC, range(20, 27))["plateau_bytes_per_s"]
    rows = []
    for NP, D in [(8, 1), (256, 30), (512, 30), (128, 100), (1024, 100), (4096, 300), (4096, 1024)]:
        if NP > max_np_for(D, ACC):
            continue
        x = torch.rand(NP, D, device=ACC)
        fn = lambda: de_generation(x)
        wall = time_fn(fn, IS_CUDA, reps=101, warmup=10)["median_s"] * 1e6
        tr = profile_gen(fn)
        busy = tr["busy_us_per_gen"]
        row = {"NP": NP, "D": D, "N": NP * D, "wall_us": wall, "busy_us": busy,
               "busy_fraction": busy / wall if wall else None,
               "idle_us": wall - busy,
               "kernels_per_gen": tr["kernels_per_gen"],
               "launch_calls_per_gen": tr["launch_calls_per_gen"],
               "sync_calls_per_gen": tr["sync_calls_per_gen"],
               "kernel_time_traffic_B_per_el": (busy * 1e-6 * bw / (NP * D)) if bw else None,
               "top_kernels": tr["top_kernels"],
               "last_generation_kernels": tr["last_generation_kernels"]}
        rows.append(row)
        print(f"  NP={NP:<5} D={D:<5} wall {wall:8.1f} us  GPU-busy {busy:8.1f} us "
              f"({100*busy/wall:5.1f}%)  kernels/gen {tr['kernels_per_gen']:5.1f}  "
              f"launch calls/gen {tr['launch_calls_per_gen']:5.1f}")
        del x
        if IS_CUDA:
            torch.cuda.empty_cache()
    OUT["s1_trace"] = {"bw_plateau_bytes_per_s": bw, "rows": rows}

    print("\n  kernels per generation by objective (NP=256, D=30)")
    objs = []
    for name, f in OBJECTIVES:
        x = torch.rand(256, 30, device=ACC)
        fn = lambda: de_generation(x, f)
        wall = time_fn(fn, IS_CUDA, reps=101, warmup=10)["median_s"] * 1e6
        tr = profile_gen(fn)
        objs.append({"objective": name, "kernels_per_gen": tr["kernels_per_gen"],
                     "launch_calls_per_gen": tr["launch_calls_per_gen"],
                     "busy_us": tr["busy_us_per_gen"], "wall_us": wall})
        print(f"    {name:<11} kernels/gen {tr['kernels_per_gen']:5.1f}  "
              f"busy {tr['busy_us_per_gen']:7.1f} us  wall {wall:7.1f} us")
    OUT["s1_objectives"] = objs


# ============================================================ S2 counters
CHILD = r'''
import torch
def sphere(z): return (z*z).sum(dim=1)
def gen(x):
    NP, D = x.shape; dev = x.device
    idx = torch.randint(0, NP, (3, NP), device=dev)
    v = x[idx[0]] + 0.5 * (x[idx[1]] - x[idx[2]])
    mask = torch.rand(NP, D, device=dev) < 0.9
    mask.scatter_(1, torch.randint(0, D, (NP, 1), device=dev), True)
    u = torch.where(mask, v, x)
    better = (sphere(u) < sphere(x)).unsqueeze(1)
    return torch.where(better, u, x)
x = torch.rand(4096, 1024, device="cuda")
for _ in range(3): gen(x)
torch.cuda.synchronize()
torch.cuda.profiler.start()
gen(x)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
'''


CHILD_CUPTI = CHILD.split("x = torch.rand")[0] + r'''
import json, os, tempfile
from torch.profiler import profile, ProfilerActivity
from torch._C._profiler import _ExperimentalConfig
out = {}
try:
    cfg = _ExperimentalConfig(profiler_metrics=["dram__bytes_read.sum", "dram__bytes_write.sum"],
                              profiler_measure_per_kernel=True)
    x = torch.rand(4096, 1024, device="cuda")
    for _ in range(3): gen(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], experimental_config=cfg) as prof:
        gen(x); torch.cuda.synchronize()
    fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    prof.export_chrome_trace(path)
    ev = json.load(open(path))["traceEvents"]; os.remove(path)
    tot, hits = 0.0, 0
    for e in ev:
        a = e.get("args", {})
        for k in ("dram__bytes_read.sum", "dram__bytes_write.sum"):
            if k in a:
                tot += float(a[k]); hits += 1
    out = {"counter_values_found": hits}
    if hits:
        out["dram_bytes_per_element"] = tot / x.numel()
except Exception as exc:
    out = {"error": f"{type(exc).__name__}: {exc}"[:500]}
print("RESULT " + json.dumps(out))
'''


def find_tool(name):
    p = shutil.which(name)
    if p:
        return p
    for pat in [f"/usr/local/cuda/bin/{name}", f"/usr/local/cuda*/bin/{name}",
                f"/opt/nvidia/nsight-compute/*/{name}", f"/usr/local/cuda*/nsight-compute*/{name}"]:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def parse_counter_csv(txt):
    """Rows of (kernel, metric, bytes) from ncu or nvprof --csv output."""
    import csv, io
    lines = [l for l in txt.splitlines() if l.startswith('"')]
    out, header = [], None
    for rec in csv.reader(io.StringIO("\n".join(lines))):
        if "Metric Name" in rec:
            header = rec
            continue
        if not header or len(rec) != len(header):
            continue
        r = dict(zip(header, rec))
        metric = r.get("Metric Name", "")
        if "dram" not in metric:
            continue
        unit = r.get("Metric Unit", "byte")
        raw = r.get("Metric Value", r.get("Avg", "")).replace(",", "")
        m = re.match(r"^([0-9.eE+-]+)\s*([A-Za-z]*)$", raw.strip())
        if not m:
            continue
        val = float(m.group(1))
        unit = m.group(2) or unit
        scale = {"byte": 1, "B": 1, "Kbyte": 1e3, "KB": 1e3, "Mbyte": 1e6, "MB": 1e6,
                 "Gbyte": 1e9, "GB": 1e9}.get(unit, 1)
        inv = float(r.get("Invocations", "1") or 1)
        out.append({"kernel": r.get("Kernel Name", r.get("Kernel", ""))[:90], "metric": metric,
                    "bytes": val * scale * inv})
    return out


def s2_counters():
    res = {"N": 4096 * 1024}
    if not IS_CUDA:
        OUT["s2_counters"] = {"skipped": "dry run"}
        return
    fd, child = tempfile.mkstemp(suffix=".py")
    os.write(fd, CHILD.encode())
    os.close(fd)
    ncu, nvprof = find_tool("ncu"), find_tool("nvprof")
    res["ncu_path"], res["nvprof_path"] = ncu, nvprof
    print(f"  ncu: {ncu}   nvprof: {nvprof}")
    attempts = []
    if ncu:
        attempts.append(("ncu", [ncu, "--profile-from-start", "off", "--print-units", "base",
                                 "--metrics", "dram__bytes_read.sum,dram__bytes_write.sum",
                                 "--csv", sys.executable, child]))
    if nvprof:
        attempts.append(("nvprof", [nvprof, "--profile-from-start", "off",
                                    "--metrics", "dram_read_bytes,dram_write_bytes",
                                    "--csv", sys.executable, child]))
    for name, cmd in attempts:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            txt = p.stdout + "\n" + p.stderr
            refused = any(k in txt for k in ("ERR_NVGPUCTRPERM", "ermission", "not supported",
                                             "INSUFFICIENT_PRIVILEGES", "doesn't support"))
            rows = parse_counter_csv(txt)
            total = sum(r["bytes"] for r in rows)
            res[name] = {"returncode": p.returncode, "refused": refused, "kernel_rows": rows,
                         "output_head": txt[:3000], "output_tail": txt[-3000:]}
            if total > 0:
                res[name]["dram_bytes_total"] = total
                res[name]["dram_bytes_per_element"] = total / res["N"]
                print(f"  {name}: {len(rows)} counter rows, "
                      f"{total/res['N']:.1f} DRAM bytes per element per generation")
            else:
                print(f"  {name}: return code {p.returncode}, no counter values"
                      f"{' (counter access refused by the platform)' if refused else ''}")
        except Exception as exc:
            res[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  {name}: {type(exc).__name__}: {exc}")
    # CUPTI range-profiler metrics through torch.profiler (experimental API), isolated
    # in a child process so that a driver-level failure cannot end the main run
    fd, child2 = tempfile.mkstemp(suffix=".py")
    os.write(fd, CHILD_CUPTI.encode())
    os.close(fd)
    try:
        p = subprocess.run([sys.executable, child2], capture_output=True, text=True, timeout=600)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        res["cupti_metrics"] = json.loads(line[-1][7:]) if line else {
            "returncode": p.returncode, "output_tail": (p.stdout + p.stderr)[-3000:]}
        print(f"  CUPTI metrics: {res['cupti_metrics'] if 'dram_bytes_per_element' in res['cupti_metrics'] else 'no counter values returned'}")
    except Exception as exc:
        res["cupti_metrics"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  CUPTI metrics: {type(exc).__name__}: {str(exc)[:160]}")
    os.remove(child2)
    os.remove(child)
    OUT["s2_counters"] = res


# ============================================================ S3 traffic
def s3_traffic():
    acc_bw = stream_bw(ACC, range(22, 29) if IS_CUDA else range(20, 24))
    cpu_bw = stream_bw("cpu", range(22, 27) if not DRY else range(20, 23))
    print(f"  achieved bandwidth: accelerator {acc_bw['plateau_bytes_per_s']/1e9:.1f} GB/s, "
          f"host {cpu_bw['plateau_bytes_per_s']/1e9:.1f} GB/s")
    Ds = [256, 512, 1024, 2048, 3072, 4096] if not DRY else [64, 128, 256]
    NP = 4096 if not DRY else 512
    rows = []
    for D in Ds:
        for lab, dev in (("acc", ACC), ("cpu", "cpu")):
            if NP > max_np_for(D, dev):
                continue
            x = torch.rand(NP, D, device=dev)
            st = time_fn(lambda: de_generation(x), dev == "cuda", reps=11, warmup=3)
            rows.append({"device": lab, "NP": NP, "D": D,
                         "N": NP * D, "time_s": st["median_s"]})
            print(f"    {lab:<4} N={NP*D:>10,}  {st['median_s']*1e3:9.3f} ms")
            del x
            if dev == "cuda":
                torch.cuda.empty_cache()
    out = {"acc_bw": acc_bw, "cpu_bw": cpu_bw, "rows": rows}
    for dev, bw in (("acc", acc_bw["plateau_bytes_per_s"]), ("cpu", cpu_bw["plateau_bytes_per_s"])):
        pts = [r for r in rows if r["device"] == dev and r["N"] >= (4 * 2 ** 20 if not DRY else 0)]
        if len(pts) >= 3 and bw:
            N = np.array([r["N"] for r in pts], float)
            t = np.array([r["time_s"] for r in pts], float)
            s, c = np.polyfit(N, t, 1)
            out[f"{dev}_slope_s_per_el"] = float(s)
            out[f"{dev}_bytes_per_el"] = float(s * bw)
            print(f"  {dev}: slope {s*1e9:.4f} ns/element -> {s*bw:.1f} bytes/element at achieved bandwidth")
    OUT["s3_traffic"] = out


# =========================================================== S4 small NP
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


def s4_small_np():
    if not IS_CUDA:
        OUT["s4_small_np"] = {"skipped": "dry run"}
        return
    # correctness gate, same session
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
    ks = sstats.ks_2samp(eager, graphed) if HAVE_SCIPY else None
    gate = {"ks": float(ks.statistic) if ks else None, "p": float(ks.pvalue) if ks else None}
    print(f"  correctness gate: KS p = {gate['p']:.3f}")
    rows = []
    for D in [10, 30, 100, 500]:
        for NP in [1, 2, 3, 4, 8, 16]:
            g, sx = build_graph(NP, D)
            xc = torch.rand(NP, D)
            sc, sa = interleaved(lambda: de_generation(xc), lambda: g.replay(), 120, rounds=6)
            rows.append({"D": D, "NP": NP, "cpu_us": sc["median_s"] * 1e6,
                         "graph_us": sa["median_s"] * 1e6,
                         "gpu_faster": sa["median_s"] < sc["median_s"],
                         "ratio_cpu_over_graph": sc["median_s"] / sa["median_s"]})
            print(f"    D={D:<4} NP={NP:<3} host {sc['median_s']*1e6:7.1f} us  "
                  f"graph {sa['median_s']*1e6:7.1f} us  "
                  f"{'GPU faster' if sa['median_s'] < sc['median_s'] else 'host faster'}")
            del g, sx
            torch.cuda.empty_cache()
    OUT["s4_small_np"] = {"gate": gate, "rows": rows}


# ======================================================== S5 high reps
def s5_high_reps():
    s0 = cpu_times()
    g_cpu = grid(make_de, "cpu", HIGH_REPS)
    g_acc = grid(make_de, ACC, HIGH_REPS)
    fc, fa = fit_nnls(g_cpu), fit_nnls(g_acc)
    print(f"  host:        c={fc['c_s']*1e6:8.2f} us  a={fc['a_s']*1e9:8.2f} ns  "
          f"b={fc['b_s']*1e9:8.4f} ns  R2={fc['r2']:.4f}")
    print(f"  accelerator: c={fa['c_s']*1e6:8.2f} us  a={fa['a_s']*1e9:8.2f} ns  "
          f"b={fa['b_s']*1e9:8.4f} ns  R2={fa['r2']:.4f}")
    rows = []
    for D in BE_DIMS:
        p = predict(fc, fa, D)
        reps_found = []
        for k in range(BE_REPEATS):
            m = breakeven(D, make_de, HIGH_REPS)
            m["repeat"] = k
            m["np_star_predicted"] = p
            rows.append(m)
            if m.get("np_star") and not m.get("censored"):
                reps_found.append(m["np_star"])
        if reps_found and p:
            med = float(np.median(reps_found))
            print(f"    D={D:<4} measured {reps_found}  (spread {100*(max(reps_found)-min(reps_found))/med:4.1f}%)"
                  f"  predicted {p:8.1f}  error vs median {100*(p-med)/med:+6.1f}%")
        else:
            print(f"    D={D:<4} {rows[-1]}")
    OUT["s5_high_reps"] = {"reps": HIGH_REPS, "rounds": ROUNDS, "grid_cpu": g_cpu,
                           "grid_acc": g_acc, "fit_cpu": fc, "fit_acc": fa, "breakeven": rows,
                           "steal_fraction": steal_fraction(s0, cpu_times())}


# ====================================================== S7 objectives
def s7_objectives():
    """Cost grid, NNLS fit and break-even for each objective, so the model can be
    tested both a priori (Sphere coefficients, objective only in b) and with each
    objective's own coefficients."""
    out = {}
    dims = [10, 30, 100] if not DRY else [10, 100]
    for name, f in OBJECTIVES:
        maker = (lambda f_: (lambda NP, D, dev: (lambda x: (lambda: de_generation(x, f_)))(
            torch.rand(NP, D, dtype=DTYPE, device=dev))))(f)
        gc, ga = grid(maker, "cpu", 15), grid(maker, ACC, 15)
        fc, fa = fit_nnls(gc), fit_nnls(ga)
        rows = []
        for D in dims:
            m = breakeven(D, maker, 30)
            m["np_star_predicted_own"] = predict(fc, fa, D)
            rows.append(m)
        out[name] = {"grid_cpu": gc, "grid_acc": ga, "fit_cpu": fc, "fit_acc": fa, "breakeven": rows}
        print(f"  {name:<11} host c={fc['c_s']*1e6:7.1f} us a={fc['a_s']*1e9:7.1f} ns b={fc['b_s']*1e9:7.3f} ns | "
              f"acc c={fa['c_s']*1e6:7.1f} us | NP* " +
              ", ".join(f"D={r['D']}:{r.get('np_star')}/{(r.get('np_star_predicted_own') or 0):.0f}" for r in rows))
    OUT["s7_objectives"] = out


# ========================================================= S6 signature
def s6_signature():
    out = {}
    for name in ("de", "jde", "pso"):
        maker = ALGOS[name]
        fc, fa = fit_nnls(grid(maker, "cpu", 15)), fit_nnls(grid(maker, ACC, 15))
        rows = []
        for D in SIG_DIMS:
            m = breakeven(D, maker, 30, floor=16)
            m["np_star_predicted"] = predict(fc, fa, D)
            rows.append(m)
        pts = [(r["D"], r["np_star"]) for r in rows if r.get("np_star") and not r.get("censored")]
        r2 = None
        if len(pts) >= 3:
            Dv = np.array([p[0] for p in pts], float)
            inv = 1.0 / np.array([p[1] for p in pts], float)
            co = np.polyfit(Dv, inv, 1)
            pred = np.polyval(co, Dv)
            r2 = float(1 - ((inv - pred) ** 2).sum() / ((inv - inv.mean()) ** 2).sum())
        out[name] = {"fit_cpu": fc, "fit_acc": fa, "breakeven": rows, "signature_r2": r2,
                     "n_points": len(pts)}
        print(f"  {name:<4} points {len(pts)}  signature R2 = {r2}")
    OUT["s6_signature"] = out


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
    print("=" * 78 + "\n  REVISION EXPERIMENT (paper 2095)" + ("  [DRY RUN ON CPU]" if DRY else "") + "\n" + "=" * 78)
    if not DRY:
        if not torch.cuda.is_available():
            print("  ERROR: no GPU. Settings -> Accelerator -> GPU, then restart.")
            sys.exit(1)
        try:
            t = torch.ones(1024, device="cuda")
            float((t * 2).sum())
        except Exception as exc:
            print(f"  PREFLIGHT FAILED: {exc}\n  Pin torch==2.3.1+cu118 and RESTART the session.")
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
        env.update({"gpu_name": p.name, "gpu_capability": f"{p.major}.{p.minor}",
                    "gpu_mem_bytes": p.total_memory})
    for k in ("cpu_model", "gpu_name", "torch", "cuda"):
        print(f"  {k:<10} {env.get(k)}")
    OUT["environment"] = env
    tag = "".join(c if c.isalnum() else "_" for c in env.get("gpu_name", "cpu_dry_run")).strip("_")
    path = f"revision_{tag}.json"
    log = Tee(f"revision_{tag}.log")
    log.f.write(f"(log started; environment {json.dumps(env, default=str)})\n")
    sys.stdout = log
    save(path)
    section("S1 KERNEL TRACE (kernels, GPU-busy fraction)", path, s1_trace)
    section("S3 TIMING-DERIVED DRAM TRAFFIC AT LARGE N", path, s3_traffic)
    section("S4 GRAPH REPLAY vs HOST AT NP = 1..16", path, s4_small_np)
    section("S5 HIGH-REPETITION INTERLEAVED RE-MEASUREMENT", path, s5_high_reps)
    section("S7 OBJECTIVE-COST SWEEP WITH NNLS", path, s7_objectives)
    section("S6 SIX-DIMENSION SIGNATURE FOR DE, jDE, PSO", path, s6_signature)
    section("S2 HARDWARE DRAM COUNTERS (attempt)", path, s2_counters)
    print("\n" + "=" * 78 + f"\n  WROTE {path}")
    if "errors" in OUT:
        print(f"  sections with errors: {list(OUT['errors'])}  (partial data saved; send the file)")
    print("=" * 78)


if __name__ == "__main__":
    main()

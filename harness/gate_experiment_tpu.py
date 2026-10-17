#!/usr/bin/env python3
"""
================================================================================
 GATE EXPERIMENT -- TPU / XLA  (JAX)
================================================================================
 THE PREDICTION, STATED BEFORE MEASURING

 On GPU we established t = c + a*NP + b*NP*D, and that the CPU/GPU break-even
 is governed by dc = c_gpu - c_cpu. Eager PyTorch pays ~313 us per generation
 in launch and dispatch; capturing the generation into a CUDA graph cut that
 to 101 us (T4) and 63 us (P100), and on P100 c_graph fell BELOW c_cpu, which
 made the break-even vanish -- as the model said it would, in advance.

 XLA removes per-kernel launch overhead by construction: the whole generation
 is compiled and fused into one executable. So the model predicts:

   (P1) c_tpu should resemble the GRAPHED GPU case, not the eager one.
   (P2) the break-even should therefore be very low, or absent entirely.
   (P3) a fori_loop fusing G generations into ONE dispatch should reduce the
        per-generation fixed cost by roughly a further factor of G, because
        the host round trip is amortised.

 These are falsifiable. If c_tpu comes out near 300 us, the model's account of
 what c represents is wrong, and that is a result worth reporting too.

 THREE XLA-SPECIFIC HAZARDS, HANDLED HERE

  1. LAZINESS. JAX dispatches asynchronously. Timing without
     block_until_ready() measures queueing, not execution. Every timed region
     below blocks explicitly.

  2. RECOMPILATION. XLA compiles per input SHAPE. The 9x6 grid triggers 54
     compilations. Compile time is measured separately and excluded from all
     reported timings; it is reported because it is itself a real cost of the
     TPU path that a practitioner pays.

  3. DONATION / ALIASING. Repeated calls on the same buffers can alias in ways
     that change timing. We keep inputs live and take fresh outputs.

 BASELINE FAIRNESS
   The CPU baseline runs JAX on the TPU VM's own host CPU, so the framework is
   held constant and only the device changes. This host is NOT the Xeon from
   the GPU sessions, which is deliberate -- it also gives the second CPU the
   study needs.

 KAGGLE: Accelerator -> TPU VM v5e-8 (or v3-8). Writes gate_tpu_<device>.json.
================================================================================
"""
import json, os, platform, sys, time
from datetime import datetime, timezone
from functools import partial

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print("JAX not found. On a Kaggle TPU session it is preinstalled.")
    print("If missing:  !pip install -q jax[tpu] -f "
          "https://storage.googleapis.com/jax-releases/libtpu_releases.html")
    sys.exit(1)

SEED        = 20260820
DTYPE       = jnp.float32
DIMENSIONS  = [10, 30, 50, 100, 200, 500]
GRID_NP     = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
GRID_D      = [2, 5, 10, 30, 100, 300]

C_PROBE_NP, C_PROBE_D = 8, 1
REPS, WARMUP          = 31, 10
C_PROBE_REPS          = 101
NP_FLOOR              = 4          # lowered: TPU break-even is expected to be tiny
FUSE_G                = 32         # generations fused into one dispatch


# ============================================================== environment
def capture_environment():
    devs = jax.devices()
    env = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "jax": jax.__version__,
           "jax_backend": jax.default_backend(),
           "device_count": jax.device_count(),
           "local_device_count": jax.local_device_count(),
           "devices": [str(d) for d in devs],
           "device_kind": devs[0].device_kind if devs else None,
           "cpu_count": os.cpu_count(), "seed": SEED,
           "harness": "tpu-v1"}
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    env["cpu_model"] = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    return env


def tag(env):
    k = env.get("device_kind") or env.get("jax_backend") or "unknown"
    return "".join(c if c.isalnum() else "_" for c in str(k)).strip("_")


# ==================================================================== kernel
def sphere(z):
    return jnp.sum(z * z, axis=1)


# ------------------------------------------------------------------ NumPy CPU
# The v1 run used JAX-on-CPU as the baseline. That was wrong. JAX's CPU backend
# carries large per-call dispatch overhead and its timings came out
# NON-MONOTONIC in NP (D=5: 105.4 -> 94.0 -> 71.6 us as NP increased), which is
# physically impossible. The fit absorbed that jitter: R2 fell to 0.9397 and
# fitted c_cpu (288 us) exceeded c_tpu (237 us), making dc negative so no
# prediction could be formed. The study's own methodological claim is that
# accelerators must be compared against a PROPERLY OPTIMISED CPU baseline, and
# JAX-CPU is not one. NumPy is, and it matches the vectorised-CPU baseline used
# in the GPU sessions.

def np_sphere(z):
    return np.sum(z * z, axis=1)


def np_de_generation(x, rng):
    NP, D = x.shape
    idx = rng.integers(0, NP, size=(3, NP))
    v = x[idx[0]] + 0.5 * (x[idx[1]] - x[idx[2]])
    mask = rng.random((NP, D)) < 0.9
    mask[np.arange(NP), rng.integers(0, D, size=NP)] = True
    u = np.where(mask, v, x)
    better = (np_sphere(u) < np_sphere(x))[:, None]
    return np.where(better, u, x)


def time_numpy(NP, D, reps=REPS, warmup=WARMUP):
    rng = np.random.default_rng(SEED)
    x = rng.random((NP, D), dtype=np.float32)
    for _ in range(warmup):
        np_de_generation(x, rng)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        np_de_generation(x, rng)
        samples.append(time.perf_counter() - t0)
    st = _sum(samples); st["compile_s"] = 0.0
    return st["median_s"], st


def _de_step(x, key):
    NP, D = x.shape
    k1, k2, k3 = jax.random.split(key, 3)
    idx = jax.random.randint(k1, (3, NP), 0, NP)
    v = x[idx[0]] + 0.5 * (x[idx[1]] - x[idx[2]])
    mask = jax.random.uniform(k2, (NP, D)) < 0.9
    jr = jax.random.randint(k3, (NP,), 0, D)
    mask = mask.at[jnp.arange(NP), jr].set(True)
    u = jnp.where(mask, v, x)
    better = (sphere(u) < sphere(x))[:, None]
    return jnp.where(better, u, x)


@jax.jit
def de_generation(x, key):
    """One generation, one dispatch. Analogue of eager PyTorch."""
    return _de_step(x, key)


@partial(jax.jit, static_argnums=(2,))
def de_fused(x, key, n_gen):
    """
    n_gen generations fused into a SINGLE dispatch via lax.fori_loop.
    Analogue of CUDA graph capture: the host round trip is paid once.
    """
    def body(i, carry):
        xx, kk = carry
        kk, sub = jax.random.split(kk)
        return (_de_step(xx, sub), kk)

    x_out, _ = jax.lax.fori_loop(0, n_gen, body, (x, key))
    return x_out


# =================================================================== timing
def _sum(s):
    a = np.asarray(s, float)
    return {"median_s": float(np.median(a)), "mean_s": float(a.mean()),
            "std_s": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min_s": float(a.min()), "p05_s": float(np.percentile(a, 5)),
            "p95_s": float(np.percentile(a, 95)),
            "iqr_s": float(np.percentile(a, 75) - np.percentile(a, 25)),
            "n": int(a.size)}


def timed(fn, warmup=WARMUP, reps=REPS):
    """
    fn() must return a JAX array. We block on it so we time execution rather
    than dispatch. The FIRST call compiles; that cost is captured separately
    and excluded from the reported statistics.
    """
    t0 = time.perf_counter()
    out = fn()
    out.block_until_ready()
    compile_s = time.perf_counter() - t0

    for _ in range(warmup):
        fn().block_until_ready()

    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn().block_until_ready()
        samples.append(time.perf_counter() - t0)
    st = _sum(samples)
    st["compile_s"] = compile_s
    return st


def make_pop(NP, D, device, seed=0):
    key = jax.random.PRNGKey(seed)
    x = jax.random.uniform(key, (NP, D), dtype=DTYPE)
    return jax.device_put(x, device)


def time_generation(device, NP, D, reps=REPS, warmup=WARMUP):
    x = make_pop(NP, D, device)
    key = jax.device_put(jax.random.PRNGKey(1), device)
    st = timed(lambda: de_generation(x, key), warmup=warmup, reps=reps)
    return st["median_s"], st


def time_fused(device, NP, D, n_gen=FUSE_G, reps=7, warmup=3):
    x = make_pop(NP, D, device)
    key = jax.device_put(jax.random.PRNGKey(1), device)
    st = timed(lambda: de_fused(x, key, n_gen), warmup=warmup, reps=reps)
    return st["median_s"] / n_gen, st


# ============================================================ measurements
def bandwidth_curve(device, label):
    """
    STREAM triad on the accelerator. v1 omitted this, so the b-vs-bandwidth
    comparison had to use the TPU SPEC figure (819 GB/s) against MEASURED
    figures for the GPUs -- not a fair comparison. Measure it.
    """
    print(f"\n  [{label}] STREAM triad (2 reads + 1 write per element)")
    rows = []
    for k in range(20, 27):
        n = 2 ** k
        try:
            key = jax.random.PRNGKey(k)
            b = jax.device_put(jax.random.uniform(key, (n,), dtype=DTYPE), device)
            c = jax.device_put(jax.random.uniform(key, (n,), dtype=DTYPE), device)
            f = jax.jit(lambda bb, cc: bb + 3.14159 * cc)
            st = timed(lambda: f(b, c), warmup=3, reps=9)
            bw = 12 * n / st["median_s"]
            rows.append({"n": n, "bw_bytes_per_s": bw})
            print(f"      2^{k:<2}  {bw/1e9:8.2f} GB/s")
        except Exception:
            print(f"      2^{k:<2}  skipped")
            break
    if not rows:
        return {"curve": [], "peak_bytes_per_s": None, "plateau_bytes_per_s": None}
    peak = max(r["bw_bytes_per_s"] for r in rows)
    print(f"      peak {peak/1e9:.2f} GB/s  plateau {rows[-1]['bw_bytes_per_s']/1e9:.2f} GB/s")
    return {"curve": rows, "peak_bytes_per_s": peak,
            "plateau_bytes_per_s": rows[-1]["bw_bytes_per_s"]}


def fusion_sweep(device, NP=256, D=30):
    """
    Sweep the fusion factor G. If c is per-SYNCHRONISATION cost, per-generation
    fixed cost should fall as ~1/G. This is the direct test of the mechanism
    the TPU run revealed, and it discriminates that account from a per-launch
    or per-kernel one.
    """
    print(f"\n  [fusion sweep] NP={NP}, D={D}")
    rows = []
    for G in (1, 2, 4, 8, 16, 32, 64):
        try:
            x = make_pop(NP, D, device)
            key = jax.device_put(jax.random.PRNGKey(1), device)
            st = timed(lambda: de_fused(x, key, G), warmup=3, reps=7)
            per = st["median_s"] / G
            rows.append({"G": G, "total_s": st["median_s"], "per_gen_s": per})
            print(f"      G={G:<3} {per*1e6:9.2f} us/gen   total {st['median_s']*1e6:9.1f} us")
        except Exception as exc:
            print(f"      G={G:<3} failed: {type(exc).__name__}")
            break
    if len(rows) >= 3:
        G = np.array([r["G"] for r in rows], float)
        t = np.array([r["per_gen_s"] for r in rows], float)
        sl, ic = np.polyfit(np.log(G), np.log(t), 1)
        print(f"      log-log slope = {sl:.3f}   (-1.000 would be exact 1/G)")
        return {"rows": rows, "loglog_slope": float(sl)}
    return {"rows": rows}


def probe_c(device, label):
    print(f"\n  [{label}] fixed-cost probe (NP={C_PROBE_NP}, D={C_PROBE_D})")
    t, st = time_generation(device, C_PROBE_NP, C_PROBE_D,
                            reps=C_PROBE_REPS, warmup=25)
    print(f"      c = {t*1e6:9.2f} us   "
          f"[p05 {st['p05_s']*1e6:.2f}, p95 {st['p95_s']*1e6:.2f}]   "
          f"compile {st['compile_s']*1e3:.0f} ms")
    return t, st


def cost_grid(device, label, use_numpy=False):
    print(f"\n  [{label}] cost grid"
          + ("" if use_numpy else "  (each shape triggers an XLA compile)"))
    rows, compile_total = [], 0.0
    for D in GRID_D:
        line = []
        for NP in GRID_NP:
            try:
                if use_numpy:
                    t, st = time_numpy(NP, D, reps=11, warmup=5)
                else:
                    t, st = time_generation(device, NP, D, reps=11, warmup=5)
            except Exception as exc:
                line.append("   ----")
                continue
            compile_total += st["compile_s"]
            rows.append({"NP": NP, "D": D, "n_elements": NP * D,
                         "time_s": t, "timing": st})
            line.append(f"{t*1e6:7.1f}")
        print(f"      D={D:<4} " + " ".join(line))
    print(f"      total compile time excluded: {compile_total:.1f} s")
    return rows, compile_total


def fit_full(rows):
    NP = np.array([r["NP"] for r in rows], float)
    D = np.array([r["D"] for r in rows], float)
    t = np.array([r["time_s"] for r in rows], float)
    A = np.column_stack([np.ones_like(NP), NP, NP * D])
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    pred = A @ coef
    ss_res = float(((t - pred) ** 2).sum()); ss_tot = float(((t - t.mean()) ** 2).sum())
    return {"c_s": float(coef[0]), "a_s_per_individual": float(coef[1]),
            "b_s_per_element": float(coef[2]),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "n_points": len(rows)}


def measure_breakeven(D, cpu, acc, np_floor=NP_FLOOR, np_cap=2 ** 18):
    def wins(NP):
        tc, _ = time_numpy(NP, D, reps=9, warmup=3)
        ta, _ = time_generation(acc, NP, D, reps=9, warmup=3)
        return ta < tc, tc, ta

    w, tc, ta = wins(np_floor)
    if w:
        return {"D": D, "np_star": np_floor, "n_star": np_floor * D,
                "bracket": "accelerator wins at NP floor", "censored": True,
                "t_cpu_s": tc, "t_acc_s": ta}
    try:
        if not wins(np_cap)[0]:
            return {"D": D, "np_star": None, "bracket": "no crossover below cap",
                    "np_cap": np_cap}
    except Exception as exc:
        return {"D": D, "np_star": None, "bracket": f"cap failed: {exc}"}

    lo, hi = np_floor, np_cap
    import math
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
    _, tc, ta = wins(hi)
    return {"D": D, "np_star": hi, "n_star": hi * D, "censored": False,
            "t_cpu_s": tc, "t_acc_s": ta}


# ======================================================================= main
def main():
    print("=" * 78)
    print("  GATE EXPERIMENT -- TPU / XLA")
    print("=" * 78)
    env = capture_environment()
    for k in ("jax", "jax_backend", "device_kind", "device_count", "cpu_model"):
        print(f"  {k:<16} {env.get(k)}")

    if env["jax_backend"] == "cpu":
        print("\n  ERROR: JAX sees no accelerator. Set Accelerator -> TPU VM.")
        sys.exit(1)

    cpu_dev = jax.devices("cpu")[0]
    acc_dev = jax.devices()[0]
    print(f"\n  host device : {cpu_dev}")
    print(f"  accel device: {acc_dev}")

    res = {"environment": env,
           "config": {"seed": SEED, "grid_np": GRID_NP, "grid_d": GRID_D,
                      "dimensions": DIMENSIONS, "np_floor": NP_FLOOR,
                      "fuse_g": FUSE_G, "reps": REPS}}

    print("\n" + "-" * 78 + "\n  (1) FIXED COST\n" + "-" * 78)
    print("\n  [CPU/NumPy] fixed-cost probe (NP=8, D=1)")
    c_cpu, c_cpu_st = time_numpy(C_PROBE_NP, C_PROBE_D, reps=C_PROBE_REPS, warmup=25)
    print(f"      c = {c_cpu*1e6:9.2f} us   "
          f"[p05 {c_cpu_st['p05_s']*1e6:.2f}, p95 {c_cpu_st['p95_s']*1e6:.2f}]")
    c_acc, c_acc_st = probe_c(acc_dev, "TPU")
    res["c_direct"] = {"cpu_s": c_cpu, "acc_s": c_acc,
                       "cpu_stats": c_cpu_st, "acc_stats": c_acc_st}
    print(f"\n      dc = {(c_acc-c_cpu)*1e6:+.2f} us")
    print(f"\n      GPU reference (PyTorch eager) : c_gpu ~ 313 us")
    print(f"      GPU reference (CUDA graph)    : c_gpu ~ 63-101 us")
    print(f"      TPU measured                  : c_tpu = {c_acc*1e6:.2f} us")
    if c_acc < 150e-6:
        print("      -> consistent with P1: XLA behaves like the GRAPHED case.")
    else:
        print("      -> NOT consistent with P1. The model's account of c needs revision.")

    print("\n" + "-" * 78 + "\n  (1b) ACCELERATOR BANDWIDTH\n" + "-" * 78)
    res["bandwidth_acc"] = bandwidth_curve(acc_dev, "TPU")

    print("\n" + "-" * 78 + "\n  (2) COST GRID\n" + "-" * 78)
    grid_cpu, cc = cost_grid(cpu_dev, "CPU/NumPy", use_numpy=True)
    grid_acc, ca = cost_grid(acc_dev, "TPU")
    fit_cpu, fit_acc = fit_full(grid_cpu), fit_full(grid_acc)
    res.update({"grid": {"cpu": grid_cpu, "acc": grid_acc},
                "fit": {"cpu": fit_cpu, "acc": fit_acc},
                "compile_seconds": {"cpu": cc, "acc": ca}})
    for nm, f in (("CPU", fit_cpu), ("TPU", fit_acc)):
        print(f"\n      {nm}: t = {f['c_s']*1e6:8.2f}us "
              f"+ {f['a_s_per_individual']*1e9:8.2f} ns*NP "
              f"+ {f['b_s_per_element']*1e9:8.4f} ns*NP*D   R2={f['r2']:.4f}")

    dc = fit_acc["c_s"] - fit_cpu["c_s"]
    da = fit_cpu["a_s_per_individual"] - fit_acc["a_s_per_individual"]
    db = fit_cpu["b_s_per_element"] - fit_acc["b_s_per_element"]
    res["prediction"] = {"delta_c_s": dc, "delta_a_s": da, "delta_b_s": db}
    print(f"\n      dc={dc*1e6:.2f}us  da={da*1e9:.2f}ns  db={db*1e9:.4f}ns")

    print("\n" + "-" * 78 + "\n  (3) FUSED DISPATCH (P3)\n" + "-" * 78)
    try:
        t1, _ = time_generation(acc_dev, 256, 30)
        tf, stf = time_fused(acc_dev, 256, 30, FUSE_G)
        res["fused"] = {"single_s": t1, "per_gen_fused_s": tf,
                        "n_gen": FUSE_G, "stats": stf,
                        "reduction": t1 / tf if tf > 0 else None}
        print(f"      single dispatch  : {t1*1e6:9.2f} us/generation")
        print(f"      fused x{FUSE_G:<3}      : {tf*1e6:9.2f} us/generation")
        print(f"      reduction        : {t1/tf:6.2f}x")
    except Exception as exc:
        print(f"      fused path failed: {type(exc).__name__}: {exc}")
        res["fused_error"] = str(exc)

    print("\n" + "-" * 78 + "\n  (3b) FUSION SWEEP -- is c ~ 1/G?\n" + "-" * 78)
    res["fusion_sweep"] = fusion_sweep(acc_dev)

    print("\n" + "-" * 78 + "\n  (4) BREAK-EVEN (P2)\n" + "-" * 78)
    rows = []
    for D in DIMENSIONS:
        p = dc / (da + db * D) if (da + db * D) > 0 and dc > 0 else None
        m = measure_breakeven(D, cpu_dev, acc_dev)
        m["np_star_predicted"] = p
        if m.get("np_star"):
            if p and not m.get("censored"):
                m["rel_error"] = (p - m["np_star"]) / m["np_star"]
                print(f"      D={D:<4} meas={m['np_star']:>7,} pred={p:>9,.1f} "
                      f"err={m['rel_error']:>+7.1%}  N*={m['n_star']:>9,}")
            else:
                note = m.get("bracket", "no prediction (dc <= 0)")
                print(f"      D={D:<4} meas={m['np_star']:>7,} "
                      f"N*={m['n_star']:>9,}   [{note}]")
        else:
            print(f"      D={D:<4} {m.get('bracket','no crossover found')}")
        rows.append(m)
    res["breakeven"] = rows

    path = f"gate_tpu_{tag(env)}.json"
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    print("\n" + "=" * 78 + f"\n  WROTE {path}\n" + "=" * 78)


if __name__ == "__main__":
    main()

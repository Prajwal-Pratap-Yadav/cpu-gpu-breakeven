#!/usr/bin/env python3
"""
================================================================================
 TPU ADDENDUM -- two loose ends from the main TPU run
================================================================================
 1. BANDWIDTH NEVER PLATEAUED. The previous sweep stopped at 2^26 and was
    still climbing (52 -> 97 -> 160 -> 240 -> 357 -> 489 -> 565 GB/s). So
    565 GB/s is a LOWER BOUND, not the achieved bandwidth, and the b-vs-
    bandwidth comparison against the GPUs used an unfair number. This sweeps
    to 2^30 and reports where it actually flattens.

 2. FUSION DECOMPOSITION. total(G) = c_sync + G*t_work fitted at R2=0.9997 on
    one configuration (NP=256, D=30), giving c_sync=231.6us, t_work=37.6us.
    One point is an anecdote. This repeats it across several (NP, D) so we can
    state whether c_sync is a constant of the dispatch path (it should be) and
    t_work scales with NP*D (it should).

 Run on a Kaggle TPU session. No pip install needed. Writes tpu_addendum.json.
================================================================================
"""
import json, os, platform, sys, time
from datetime import datetime, timezone
from functools import partial

import numpy as np

try:
    import jax, jax.numpy as jnp
except ImportError:
    print("JAX not found -- use a Kaggle TPU session."); sys.exit(1)

SEED  = 20260820
DTYPE = jnp.float32
FUSE_CONFIGS = [(64, 10), (256, 30), (1024, 30), (256, 100), (1024, 300)]
FUSE_G = [1, 2, 4, 8, 16, 32, 64]
OUT = {}


def sphere(z):
    return jnp.sum(z * z, axis=1)


def _step(x, key):
    NP, D = x.shape
    k1, k2, k3 = jax.random.split(key, 3)
    idx = jax.random.randint(k1, (3, NP), 0, NP)
    v = x[idx[0]] + 0.5 * (x[idx[1]] - x[idx[2]])
    mask = jax.random.uniform(k2, (NP, D)) < 0.9
    jr = jax.random.randint(k3, (NP,), 0, D)
    mask = mask.at[jnp.arange(NP), jr].set(True)
    u = jnp.where(mask, v, x)
    return jnp.where((sphere(u) < sphere(x))[:, None], u, x)


@partial(jax.jit, static_argnums=(2,))
def de_fused(x, key, n_gen):
    def body(i, carry):
        xx, kk = carry
        kk, sub = jax.random.split(kk)
        return (_step(xx, sub), kk)
    out, _ = jax.lax.fori_loop(0, n_gen, body, (x, key))
    return out


def timed(fn, warmup=5, reps=15):
    t0 = time.perf_counter()
    fn().block_until_ready()
    compile_s = time.perf_counter() - t0
    for _ in range(warmup):
        fn().block_until_ready()
    s = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn().block_until_ready()
        s.append(time.perf_counter() - t0)
    a = np.asarray(s)
    return {"median_s": float(np.median(a)), "p05_s": float(np.percentile(a, 5)),
            "p95_s": float(np.percentile(a, 95)), "compile_s": compile_s,
            "n": int(a.size)}


def main():
    devs = jax.devices()
    env = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "jax": jax.__version__, "backend": jax.default_backend(),
           "device_kind": devs[0].device_kind, "device_count": jax.device_count(),
           "platform": platform.platform(), "numpy": np.__version__, "seed": SEED}
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    env["cpu_model"] = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    print("=" * 78 + "\n  TPU ADDENDUM\n" + "=" * 78)
    for k in ("jax", "backend", "device_kind", "cpu_model"):
        print(f"  {k:<14} {env.get(k)}")
    if env["backend"] == "cpu":
        print("\n  ERROR: no TPU. Set Accelerator -> TPU VM."); sys.exit(1)
    OUT["environment"] = env
    dev = devs[0]

    print("\n" + "-" * 78 + "\n  (1) EXTENDED BANDWIDTH SWEEP (to 2^30)\n" + "-" * 78)
    rows = []
    for k in range(20, 31):
        n = 2 ** k
        try:
            key = jax.random.PRNGKey(k)
            b = jax.device_put(jax.random.uniform(key, (n,), dtype=DTYPE), dev)
            c = jax.device_put(jax.random.uniform(key, (n,), dtype=DTYPE), dev)
            f = jax.jit(lambda bb, cc: bb + 3.14159 * cc)
            st = timed(lambda: f(b, c), warmup=3, reps=9)
            bw = 12 * n / st["median_s"]
            rows.append({"n": n, "working_set_bytes": 3 * n * 4,
                         "bw_bytes_per_s": bw})
            print(f"      2^{k:<2} {bw/1e9:8.2f} GB/s   "
                  f"working set {3*n*4/1e6:9.1f} MB")
            del b, c
        except Exception as exc:
            print(f"      2^{k:<2} stopped: {type(exc).__name__}")
            break
    if rows:
        peak = max(r["bw_bytes_per_s"] for r in rows)
        last3 = [r["bw_bytes_per_s"] for r in rows[-3:]]
        flat = (max(last3) - min(last3)) / np.mean(last3) < 0.05 if len(last3) == 3 else False
        print(f"\n      peak {peak/1e9:.2f} GB/s")
        print(f"      last three within 5%: {flat}  "
              f"-> {'PLATEAUED' if flat else 'STILL CLIMBING (still a lower bound)'}")
        OUT["bandwidth"] = {"curve": rows, "peak_bytes_per_s": peak,
                            "plateaued": bool(flat)}

    print("\n" + "-" * 78 + "\n  (2) FUSION DECOMPOSITION ACROSS CONFIGS\n" + "-" * 78)
    print("      model: total(G) = c_sync + G*t_work")
    fus = []
    for NP, D in FUSE_CONFIGS:
        try:
            key = jax.device_put(jax.random.PRNGKey(1), dev)
            x = jax.device_put(jax.random.uniform(jax.random.PRNGKey(0),
                                                  (NP, D), dtype=DTYPE), dev)
            Gs, tot = [], []
            for G in FUSE_G:
                st = timed(lambda: de_fused(x, key, G), warmup=3, reps=7)
                Gs.append(G); tot.append(st["median_s"])
            G = np.array(Gs, float); T = np.array(tot, float)
            A = np.column_stack([np.ones_like(G), G])
            (c_sync, t_work), *_ = np.linalg.lstsq(A, T, rcond=None)
            pred = A @ np.array([c_sync, t_work])
            r2 = 1 - ((T - pred) ** 2).sum() / ((T - T.mean()) ** 2).sum()
            row = {"NP": NP, "D": D, "n_elements": NP * D,
                   "c_sync_s": float(c_sync), "t_work_s": float(t_work),
                   "r2": float(r2), "G": Gs, "total_s": tot,
                   "per_gen_G1_s": tot[0], "per_gen_G64_s": tot[-1] / Gs[-1]}
            fus.append(row)
            print(f"      NP={NP:<5} D={D:<4} N={NP*D:>8,}  "
                  f"c_sync={c_sync*1e6:8.2f}us  t_work={t_work*1e6:8.2f}us  "
                  f"R2={r2:.5f}")
        except Exception as exc:
            print(f"      NP={NP:<5} D={D:<4} failed: {type(exc).__name__}: {exc}")
    OUT["fusion"] = fus

    if len(fus) >= 3:
        cs = np.array([r["c_sync_s"] for r in fus])
        tw = np.array([r["t_work_s"] for r in fus])
        ne = np.array([r["n_elements"] for r in fus], float)
        print(f"\n      c_sync across configs: mean {cs.mean()*1e6:.1f}us  "
              f"CV {cs.std(ddof=1)/cs.mean():.1%}")
        print("      (should be roughly constant -- it is a dispatch-path cost)")
        sl = np.polyfit(np.log(ne), np.log(tw), 1)[0]
        print(f"      t_work vs N: log-log slope {sl:+.3f} "
              f"(1.0 = linear in NP*D)")
        OUT["fusion_summary"] = {"c_sync_mean_s": float(cs.mean()),
                                 "c_sync_cv": float(cs.std(ddof=1) / cs.mean()),
                                 "t_work_loglog_slope": float(sl)}

    with open("tpu_addendum.json", "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)
    print("\n" + "=" * 78 + "\n  WROTE tpu_addendum.json\n" + "=" * 78)


if __name__ == "__main__":
    main()

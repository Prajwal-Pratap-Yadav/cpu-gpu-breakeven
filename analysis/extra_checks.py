"""NP log NP extension and synthetic check of the D^2/D^3 test (Section 5.6).
Usage: python3 analysis/extra_checks.py [data_dir] [out_json]"""
import json, sys
import numpy as np
from scipy.optimize import nnls, brentq

U = sys.argv[1] if len(sys.argv) > 1 else "data"
OUTF = sys.argv[2] if len(sys.argv) > 2 else "results/extra_checks.json"
ALG = {k: json.load(open(f"{U}/algos_{v}.json")) for k, v in (("T4", "Tesla_T4"), ("P100", "Tesla_P100_PCIE_16GB"))}


def design(rows, extra=False):
    NP = np.array([r["NP"] for r in rows], float); D = np.array([r["D"] for r in rows], float)
    cols = [np.ones_like(NP), NP, NP * D] + ([NP * np.log(NP)] if extra else [])
    return np.column_stack(cols), np.array([r["time_s"] for r in rows], float)


def nstar(cc, ca, D, extra):
    f = lambda n: ((cc[0] + cc[1] * n + cc[2] * n * D + (cc[3] * n * np.log(n) if extra else 0))
                   - (ca[0] + ca[1] * n + ca[2] * n * D + (ca[3] * n * np.log(n) if extra else 0)))
    g = np.geomspace(1.5, 2 ** 22, 2000); v = np.array([f(n) for n in g])
    i = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
    return brentq(f, g[i[0]], g[i[0] + 1]) if len(i) else np.nan


res = {}
for dev in ALG:
    for al, A_ in ALG[dev]["algorithms"].items():
        r = {}
        for extra in (False, True):
            Ac, tc = design(A_["cpu"]["grid"], extra); Aa, ta = design(A_["cuda"]["grid"], extra)
            cc, _ = nnls(Ac, tc); ca, _ = nnls(Aa, ta)
            e = [(nstar(cc, ca, x["D"], extra) - x["np_star"]) / x["np_star"] for x in A_["breakeven"]
                 if x.get("np_star") and not x.get("censored")]
            r["ext" if extra else "std"] = float(np.median(np.abs(e)))
            if extra:
                r["coef_nlogn_cpu"], r["coef_nlogn_acc"] = float(cc[3]), float(ca[3])
        res[f"{dev}/{al}"] = r
        print(dev, al, {k: (round(v * 100, 1) if k in ("std", "ext") else v) for k, v in r.items()})
rows = ALG["T4"]["algorithms"]["de"]["cpu"]["grid"]
NP = np.array([r["NP"] for r in rows], float); D = np.array([r["D"] for r in rows], float)
base = 145e-6 + 142e-9 * NP + 11e-9 * NP * D
g3 = 0.125 * base.max() / 300 ** 3; g2 = 0.125 * base.max() / (4096 * 300 ** 2)
rng = np.random.default_rng(20260820)


def gain(t):
    A3 = np.column_stack([np.ones_like(NP), NP, NP * D]); A5 = np.column_stack([A3, D ** 2, D ** 3])
    def r2(A):
        c = np.linalg.lstsq(A, t, rcond=None)[0]; p = A @ c
        return 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    return r2(A5) - r2(A3)


G0 = [gain(base * (1 + rng.normal(0, 0.03, len(base)))) for _ in range(500)]
S = base + g2 * NP * D ** 2 + g3 * D ** 3
G1 = [gain(S * (1 + rng.normal(0, 0.03, len(base)))) for _ in range(500)]
k = int(np.argmax(base))
syn = {"elementwise_median": float(np.median(G0)), "elementwise_p95": float(np.percentile(G0, 95)),
       "elementwise_max": float(np.max(G0)), "structured_median": float(np.median(G1)),
       "structured_p05": float(np.percentile(G1, 5)), "injected_share_at_largest": float((S[k] - base[k]) / S[k]),
       "noise_cv": 0.03, "trials": 500}
print(syn)
json.dump({"nplognp": res, "synthetic": syn}, open(OUTF, "w"), indent=1)

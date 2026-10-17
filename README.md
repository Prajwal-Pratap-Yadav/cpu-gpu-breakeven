# Synchronization, Not Bandwidth — code and data

Artifact for **"Synchronization, Not Bandwidth: A Predictive Cost Model for the CPU–GPU Break-Even in Population-Based Optimization"**
(E. Pramanick, P. P. Yadav, J. Sarraf, A. Swetapadma; ICMLDE, paper ID 2095).

Every table, figure and in-text number of the paper is computed by `analysis/` from the raw records in `data/`.
No number is typed by hand.

## Reproduce all tables and figures (any computer, no GPU, about 2 minutes)

```bash
pip install -r requirements.txt
bash run_all.sh
```

Outputs go to `results/`; the copies shipped there are what the paper uses, so `run_all.sh` should reproduce them exactly
(the bootstrap is seeded).

| Script | Output | Paper |
|---|---|---|
| `analysis/offline.py` | `results/offline_results.json` | bootstrap CIs and prediction intervals, signatures, NNLS vs OLS, bandwidth model (Tables 3–5, 8; Fig. 4) |
| `analysis/extra_checks.py` | `results/extra_checks.json` | NP·log NP term and synthetic D²/D³ test (Section 5.6) |
| `analysis/revision_analysis.py` | `results/revision_summary.json` | revision numbers: kernel traces, traffic, sessions and drift, graph capture at NP = 1–16, objective own-fit (Sections 4.4, 4.7, 5.1, 5.3, 5.5, 5.7) |
| `analysis/export_tables.py` (uses `tables_def.py`) | `results/tables/*.csv` | Tables 1–11 (+ energy table referred to in Section 5.7) |
| `analysis/figures.py` | `results/figures/*.png` | Figs. 1–5 |
| `analysis/traversal_audit.py` (needs PyTorch, CPU build is enough) | `results/traversal_audit.json` | instrumented ATen traffic, 116–119 B/element (Table 2) |

## Measurement harnesses and raw data

| Harness | Record in `data/` | Platform | Paper |
|---|---|---|---|
| `final_experiment.py` | `final_<gpu>.json` | T4, P100 | bandwidth, cost grid, break-even, repeatability, graph capture, energy |
| `algo_sweep.py` | `algos_<gpu>.json` | T4, P100 | DE, jDE, PSO, CMA-ES (Table 8) |
| `fitness_sweep.py` | `fitness_<gpu>.json` | T4, P100 | objective-cost sweep (Table 7) |
| `gate_experiment_tpu.py` | `gate_tpu_TPU_v5_lite.json` | TPU v5e | TPU grid and break-even |
| `tpu_addendum.py` | `tpu_addendum.json` | TPU v5e | TPU bandwidth, fusion decomposition (Table 10, Fig. 5) |
| `revision_experiment.py` | `revision_Tesla_T4.json` | T4 | revision 1: kernel traces, DRAM-counter attempt, large-N traffic, graph vs host at NP = 1–16, 60-repetition re-measurement, objectives with NNLS (Table 7), six-dimension signatures (Fig. 3b) |
| `revision2_experiment.py` | `revision2_Tesla_T4.json` | T4 | revision 2: graphed break-even from NP = 1, same-session interleaved calibration |

`data/logs/` holds the console logs of the two revision runs (including the profiler refusal messages).
Every record stores its software and hardware environment; all generators are seeded with `20260820`.

## Re-running a measurement on Kaggle

GPU (Accelerator **GPU T4 x2**, Internet **on**). Kaggle's default Python is too new for PyTorch 2.3.1, so create a
separate Python 3.12 with the exact build used in the paper, upload the harness file, and run it:

```bash
!pip install -q uv
!uv venv -q -p 3.12 /tmp/py312
!uv pip install -q --python /tmp/py312/bin/python torch==2.3.1 "numpy<2" "scipy<1.14" --extra-index-url https://download.pytorch.org/whl/cu118
!/tmp/py312/bin/python revision2_experiment.py      # or any other GPU harness
```

Each harness writes its JSON (and log) to the working directory and saves after every section.
The Tesla P100 used in the original runs is no longer offered by Kaggle, so the revision runs exist for the T4 only.

TPU (Accelerator **TPU v5e-8**): run `gate_experiment_tpu.py`, then `tpu_addendum.py`, with the preinstalled JAX.

## License

MIT (see `LICENSE`).

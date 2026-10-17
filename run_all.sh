#!/usr/bin/env bash
# Regenerates every table, figure and in-text number of the paper from data/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/tables results/figures
python3 analysis/offline.py data results/offline_results.json
python3 analysis/extra_checks.py data results/extra_checks.json
python3 analysis/revision_analysis.py data results/revision_summary.json
python3 analysis/export_tables.py
python3 analysis/figures.py data results/offline_results.json results/figures data/revision_Tesla_T4.json
if python3 -c "import torch" 2>/dev/null; then python3 analysis/traversal_audit.py results/traversal_audit.json
else echo "PyTorch not installed: skipped traversal_audit.py (results/traversal_audit.json is the shipped copy)"; fi
echo "done: see results/"

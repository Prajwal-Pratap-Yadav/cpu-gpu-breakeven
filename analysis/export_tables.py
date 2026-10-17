"""Writes every manuscript table as CSV (results/tables/*.csv) from the raw records.
Usage: python3 analysis/export_tables.py   (run offline.py first)"""
import csv, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tables_def as T

os.makedirs("results/tables", exist_ok=True)
clean = lambda s: re.sub(r"\$|<[^>]+>|_\{|\^\{|\}", "", str(s))
for name, t in T.TABLES.items():
    with open(f"results/tables/{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        hdr = t["header"] if isinstance(t["header"][0], list) else [t["header"]]
        for h in hdr:
            w.writerow([clean(x[0] if isinstance(x, tuple) else x) for x in h])
        for r in t["rows"]:
            w.writerow([clean(x) for x in r])
    print("results/tables/%s.csv" % name)

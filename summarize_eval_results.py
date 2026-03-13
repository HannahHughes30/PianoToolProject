from pathlib import Path
import re
import csv

log_path = Path("data/eval_run_2026-02-26/full_log.txt")
out_csv  = Path("data/eval_run_2026-02-26/summary.csv")

text = log_path.read_text(errors="ignore")

# Each block contains:
# Saved: data/<name>_gt.csv rows: <n>
# ...
# 💾 Saved predictions to: data/predictions/<name>_input_predictions.csv
# Total notes: <N>
# Exact match accuracy: <pct> %

pat = re.compile(
    r"Saved:\s+(?P<gt>data/[^ ]+_gt\.csv)\s+rows:\s+(?P<gt_rows>\d+).*?"
    r"Saved predictions to:\s+(?P<pred>data/predictions/[^ \n]+).*?"
    r"Total notes:\s+(?P<total>\d+).*?"
    r"Exact match accuracy:\s+(?P<acc>[0-9.]+)\s*%",
    re.S
)

rows = []
for m in pat.finditer(text):
    gt = m.group("gt")
    name = Path(gt).name.replace("_gt.csv", "")
    rows.append({
        "name": name,
        "gt_csv": gt,
        "pred_csv": m.group("pred"),
        "gt_rows": int(m.group("gt_rows")),
        "total_notes_reported": int(m.group("total")),
        "exact_match_acc_percent": float(m.group("acc")),
    })

rows.sort(key=lambda r: r["name"])

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
    if rows:
        w.writeheader()
        w.writerows(rows)

print(f"Wrote {out_csv} rows={len(rows)}")

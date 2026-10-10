#!/usr/bin/env python
"""Plots from MEASURED values (default: results/results_from_notebook.csv; or a CSV with the same columns).
Replaces the notebook's plotting cell, which used hand-typed numbers that did not match the printed outputs."""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/results_from_notebook.csv"
df = pd.read_csv(src)
u = df[df.config.str.startswith("unlearned")].copy()
ref = u[u.config == "unlearned_fp16"].iloc[0]
u["dNGR"] = u.ngr_percent - ref.ngr_percent
u["dCLIP"] = u.mean_clip - ref.mean_clip
u["label"] = u.config.str.replace("unlearned_", "", regex=False)
u.to_csv("results/deltas_vs_unlearned_fp16.csv", index=False)
print(u[["label", "n_positive", "ngr_percent", "mean_clip", "dNGR", "dCLIP"]].to_string(index=False))

plt.figure(figsize=(7, 5))
q = u[u.config != "unlearned_fp16"]
plt.scatter(q.dCLIP, q.dNGR, s=110, edgecolors="black", zorder=3)
# points with identical coordinates are labelled once, with all their names stacked
for (x, y), g in q.groupby([q.dCLIP.round(6), q.dNGR.round(6)]):
    plt.annotate("\n".join(g.label), (x, y), textcoords="offset points", xytext=(-8, 10), ha="right", fontsize=8)
plt.axhline(0, color="gray", ls="--"); plt.axvline(0, color="gray", ls="--")
plt.xlabel("ΔCLIP vs unlearned FP16"); plt.ylabel("ΔNGR (percentage points) vs unlearned FP16")
plt.title("Unlearned model: ΔCLIP vs ΔNGR (1 image = 0.79 pp)"); plt.tight_layout()
plt.savefig("results/figures/scatter_dclip_dngr.png", dpi=200); plt.close()

plt.figure(figsize=(9, 4.5))
bars = plt.bar(u.label, u.ngr_percent, edgecolor="black")
for b, n in zip(bars, u.n_positive):
    plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{n}/127", ha="center", fontsize=8)
plt.ylabel("NGR (%) ↓"); plt.xticks(rotation=25, ha="right"); plt.title("Unlearned model NGR by configuration")
plt.tight_layout(); plt.savefig("results/figures/bar_ngr.png", dpi=200); plt.close()
print("figures written")

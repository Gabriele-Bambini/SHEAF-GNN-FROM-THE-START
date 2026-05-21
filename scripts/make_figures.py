"""
Nature figure pipeline (Phase C).

Reads the JSON/parquet artifacts produced by the evaluation pipeline and
emits publication-ready PDFs + the LaTeX table cells. Run AFTER:
    scripts/run_ablations.py       → results/ablation_table.tex
    scripts/run_benchmarks.py      → results/benchmarks.json
    scripts/run_pipeline.py        → results/brca_heldout.json

Figures produced (each in results/figures/):
    fig2_train_curves.pdf          — val loss by model/seed
    fig3_zeroshot_crc.pdf          — BRCA→CRC Pearson bars
    fig4_ablation_heatmap.pdf      — stalk_dim × allele heatmap
    fig5_allele_swap.pdf           — Jaccard curves for hotspot swaps
    fig6_metabric_enrichment.pdf   — BH-corrected enrichment q-values per allele
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
ARTIFACTS_DIR = RES / "latest" if (RES / "latest").exists() else RES
FIGDIR = ARTIFACTS_DIR / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def _require_mpl():
    if plt is None:
        raise SystemExit("matplotlib not installed — `pip install matplotlib`.")


def fig_zeroshot_crc(path: Path = ARTIFACTS_DIR / "benchmarks.json"):
    _require_mpl()
    if not path.exists():
        print(f"skip fig3: {path} missing"); return
    data = json.loads(path.read_text())
    models = list(data.keys())
    means = [data[m].get("crc_pearson_mean", np.nan) for m in models]
    stds = [data[m].get("crc_pearson_std", 0.0) for m in models]
    order = np.argsort(means)[::-1]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([models[i] for i in order], [means[i] for i in order],
           yerr=[stds[i] for i in order], capsize=3, color="#4c72b0")
    ax.axhline(0.35, color="red", ls="--", lw=0.8, label="P1 threshold")
    ax.set_ylabel("BRCA→CRC zero-shot Pearson r")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGDIR / "fig3_zeroshot_crc.pdf"
    fig.savefig(out); print(f"wrote {out}")


def fig_allele_swap(path: Path = ARTIFACTS_DIR / "allele_swap.json"):
    _require_mpl()
    if not path.exists():
        print(f"skip fig5: {path} missing"); return
    data = json.loads(path.read_text())
    fig, ax = plt.subplots(figsize=(5, 3))
    for swap, curve in data.items():
        ks = sorted(int(k.split("_")[-1]) for k in curve if k.startswith("jaccard_topk_"))
        ys = [curve[f"jaccard_topk_{k}"] for k in ks]
        ax.plot(ks, ys, marker="o", label=swap)
    ax.axhline(0.30, color="red", ls="--", lw=0.8, label="P2 threshold")
    ax.set_xscale("log")
    ax.set_xlabel("k (top-k genes)")
    ax.set_ylabel("Jaccard distance vs allele-A baseline")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIGDIR / "fig5_allele_swap.pdf"
    fig.savefig(out); print(f"wrote {out}")


def fig_metabric_enrichment(path: Path = ARTIFACTS_DIR / "metabric_enrichment.csv"):
    _require_mpl()
    if not path.exists():
        print(f"skip fig6: {path} missing"); return
    df = pd.read_csv(path)
    df = df.sort_values("q_bh")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(df["allele"], -np.log10(df["q_bh"].clip(lower=1e-16)),
           color=["#4c72b0" if p else "#999999" for p in df["passes_p3"]])
    ax.axhline(-np.log10(0.05), color="red", ls="--", lw=0.8, label="q=0.05")
    ax.set_ylabel(r"$-\log_{10}(q_\mathrm{BH})$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGDIR / "fig6_metabric_enrichment.pdf"
    fig.savefig(out); print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["all"])
    args = ap.parse_args()

    todo = args.only
    if "all" in todo or "3" in todo: fig_zeroshot_crc()
    if "all" in todo or "5" in todo: fig_allele_swap()
    if "all" in todo or "6" in todo: fig_metabric_enrichment()


if __name__ == "__main__":
    main()

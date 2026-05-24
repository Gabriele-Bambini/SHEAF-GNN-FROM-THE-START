"""
LaTeX-ready results table generator for TS-GNN evaluation.

Produces publication-quality tables from evaluation metric dicts.
"""

from typing import Dict, List, Optional


def format_results_table(
    results: Dict[str, Dict],
    caption: str = "TS-GNN Evaluation Results",
    label: str = "tab:results",
) -> str:
    """
    Generate a LaTeX table from evaluation results.

    Args:
        results: Dict mapping metric_name -> {key: value, ...}
        caption: LaTeX table caption.
        label: LaTeX table label.

    Returns:
        LaTeX table string ready for inclusion in a paper.
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{Value} & \textbf{p-value} \\")
    lines.append(r"\midrule")

    for metric_name, values in sorted(results.items()):
        display_name = metric_name.replace("_", " ").title()

        # Extract primary value and p-value
        primary = _extract_primary_value(values)
        p_val = values.get("p_value", values.get("fisher_p", None))

        val_str = f"{primary:.4f}" if isinstance(primary, float) else str(primary)
        p_str = f"{p_val:.4e}" if isinstance(p_val, float) else "---"

        lines.append(f"  {display_name} & {val_str} & {p_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def format_ablation_table(
    ablation_results: Dict[str, Dict[str, float]],
    metric_name: str = "val_loss",
    caption: str = "Ablation Study Results",
    label: str = "tab:ablation",
) -> str:
    """
    Generate a LaTeX table for ablation studies.

    Args:
        ablation_results: Dict mapping ablation_name -> {metric: value, ...}
        metric_name: Which metric to show.
        caption: LaTeX caption.
        label: LaTeX label.

    Returns:
        LaTeX table string.
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(f"\\textbf{{Configuration}} & \\textbf{{{metric_name}}} \\\\")
    lines.append(r"\midrule")

    for name, metrics in sorted(ablation_results.items()):
        display = name.replace("_", " ").title()
        val = metrics.get(metric_name, 0.0)
        lines.append(f"  {display} & {val:.4f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def format_comparison_table(
    model_results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    caption: str = "Model Comparison",
    label: str = "tab:comparison",
) -> str:
    """
    Generate a LaTeX table comparing TS-GNN against baselines.

    Args:
        model_results: Dict mapping model_name -> {metric: value, ...}
        metrics: Which metrics to include as columns.
        caption: LaTeX caption.
        label: LaTeX label.

    Returns:
        LaTeX table string.
    """
    if metrics is None:
        # Collect all metrics from all models
        all_metrics = set()
        for m in model_results.values():
            all_metrics.update(m.keys())
        metrics = sorted(all_metrics)

    n_cols = len(metrics) + 1
    col_spec = "l" + "c" * len(metrics)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header
    header = r"\textbf{Model}"
    for m in metrics:
        header += f" & \\textbf{{{m.replace('_', ' ').title()}}}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Find best values per metric for bolding
    best_vals = {}
    for m in metrics:
        vals = [model_results[model].get(m, float("inf")) for model in model_results]
        best_vals[m] = min(vals) if "loss" in m else max(vals)

    # Rows
    for model_name, model_metrics in model_results.items():
        display = model_name.replace("_", " ")
        row = f"  {display}"
        for m in metrics:
            v = model_metrics.get(m, 0.0)
            val_str = f"{v:.4f}"
            if abs(v - best_vals[m]) < 1e-8:
                val_str = f"\\textbf{{{val_str}}}"
            row += f" & {val_str}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(f"\\end{{tabular}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _extract_primary_value(values: dict) -> float:
    """Extract the primary metric value from a results dict."""
    # Priority order for primary value
    for key in ["pearson_r", "spearman_rho", "jaccard_index", "odds_ratio",
                 "observed_stat", "effect_size", "overlap_count"]:
        if key in values:
            return values[key]
    # Fallback: first numeric value
    for v in values.values():
        if isinstance(v, (int, float)):
            return v
    return 0.0


def print_results_summary(results: Dict[str, Dict], title: str = "Evaluation Results"):
    """Print a formatted plain-text summary of evaluation results."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for metric, values in sorted(results.items()):
        display = metric.replace("_", " ").title()
        primary = _extract_primary_value(values)
        p_val = values.get("p_value", values.get("fisher_p", None))
        p_str = f"  (p={p_val:.4e})" if isinstance(p_val, float) else ""
        print(f"  {display:40s}  {primary:>10.4f}{p_str}")
    print(f"{'='*60}\n")

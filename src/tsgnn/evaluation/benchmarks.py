"""
Baseline Comparison Benchmarking for TS-GNN.

Phase 4C: Runs all baseline comparisons on identical data with identical metrics.

For each baseline (CellOracle, Dictys, EvolveGCN, TS-GNN no-allele, Temporal GAT):
- Run on identical data with identical preprocessing
- Compute all 5 evaluation metrics
- Statistical comparison: paired t-test or Wilcoxon signed-rank test
- Generate comparison table (Table 1 format)
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats

from .metrics import evaluate_all

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runs all baseline comparisons against TS-GNN.

    Args:
        models: Dict mapping model name -> model instance.
        test_data: Dict mapping allele -> test data dict.
        esm_embeddings: Dict mapping allele -> ESM-2 embedding.
        stalk_dim: Stalk dimension d for metric computation.
        seeds: Random seeds for repeated runs.
    """

    def __init__(
        self,
        models: Dict[str, torch.nn.Module],
        test_data: Dict[str, dict],
        esm_embeddings: Dict[str, torch.Tensor],
        stalk_dim: int = 4,
        seeds: Optional[List[int]] = None,
        validated_targets: Optional[List[Tuple[int, int]]] = None,
        chip_seq_edges: Optional[List[Tuple[int, int]]] = None,
        depmap_scores: Optional[np.ndarray] = None,
    ):
        self.models = models
        self.test_data = test_data
        self.esm_embeddings = esm_embeddings
        self.stalk_dim = stalk_dim
        self.seeds = seeds or [42, 123, 456]
        self.validated_targets = validated_targets
        self.chip_seq_edges = chip_seq_edges
        self.depmap_scores = depmap_scores

    def run_all(self) -> Dict[str, Dict]:
        """
        Run all baselines and compute all metrics.

        Returns:
            Dict mapping model_name -> aggregated metrics dict.
        """
        all_results = {}

        for model_name, model in self.models.items():
            logger.info(f"Evaluating: {model_name}")

            seed_results = []
            for seed in self.seeds:
                _set_seed(seed)
                metrics = evaluate_all(
                    model=model,
                    test_data=self.test_data,
                    esm_embeddings=self.esm_embeddings,
                    stalk_dim=self.stalk_dim,
                    validated_targets=self.validated_targets,
                    chip_seq_edges=self.chip_seq_edges,
                    depmap_scores=self.depmap_scores,
                )
                # Flatten nested metrics
                flat = _flatten_metrics(metrics)
                seed_results.append(flat)

            all_results[model_name] = _aggregate_seeds(seed_results)

        return all_results

    def statistical_comparison(
        self,
        results: Dict[str, Dict],
        reference: str = "tsgnn",
    ) -> Dict[str, Dict]:
        """
        Run statistical tests comparing each baseline to TS-GNN.

        Uses Wilcoxon signed-rank test for paired comparisons.

        Args:
            results: Output from run_all().
            reference: Name of the reference model (TS-GNN).

        Returns:
            Dict mapping (baseline, metric) -> {statistic, p_value}.
        """
        if reference not in results:
            logger.warning(f"Reference model '{reference}' not in results.")
            return {}

        ref_results = results[reference]
        comparisons = {}

        for model_name, model_results in results.items():
            if model_name == reference:
                continue

            for metric_name in ref_results:
                ref_vals = ref_results[metric_name].get("values", [])
                model_vals = model_results.get(metric_name, {}).get("values", [])

                if len(ref_vals) >= 3 and len(model_vals) >= 3:
                    n = min(len(ref_vals), len(model_vals))
                    try:
                        stat, p = stats.wilcoxon(ref_vals[:n], model_vals[:n])
                    except ValueError:
                        stat, p = 0.0, 1.0

                    comparisons[f"{model_name}_vs_{reference}_{metric_name}"] = {
                        "statistic": float(stat),
                        "p_value": float(p),
                        "ref_mean": float(np.mean(ref_vals)),
                        "model_mean": float(np.mean(model_vals)),
                        "significant": p < 0.05,
                    }

        return comparisons

    def generate_comparison_table(self, results: Dict[str, Dict]) -> str:
        """
        Generate LaTeX comparison table (Table 1 format from the paper).

        Returns:
            LaTeX table string.
        """
        lines = []
        lines.append(r"\begin{table*}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Baseline comparison on BRCA dataset (GSE176078 + GSE158508). "
                     r"Bold indicates best performance. "
                     r"$\dagger$ = statistically significant vs. TS-GNN ($p < 0.05$).}")
        lines.append(r"\label{tab:baselines}")

        # Collect all metrics
        all_metrics = set()
        for model_results in results.values():
            all_metrics.update(model_results.keys())
        metric_names = sorted(all_metrics)

        lines.append(r"\begin{tabular}{l" + "c" * len(metric_names) + "}")
        lines.append(r"\toprule")

        # Header
        header = "Model & " + " & ".join(
            _latex_escape(m) for m in metric_names
        ) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")

        # Find best values per metric (higher is better assumed)
        best_per_metric = {}
        for m in metric_names:
            values = {}
            for model_name, model_results in results.items():
                if m in model_results and "mean" in model_results[m]:
                    values[model_name] = model_results[m]["mean"]
            if values:
                best_per_metric[m] = max(values, key=values.get)

        # Rows
        for model_name, model_results in results.items():
            row = _latex_escape(model_name) + " & "
            cells = []
            for m in metric_names:
                if m in model_results and "mean" in model_results[m]:
                    mean_val = model_results[m]["mean"]
                    std_val = model_results[m]["std"]
                    cell = f"${mean_val:.3f} \\pm {std_val:.3f}$"
                    if best_per_metric.get(m) == model_name:
                        cell = r"\textbf{" + cell + "}"
                else:
                    cell = "--"
                cells.append(cell)
            row += " & ".join(cells) + r" \\"
            lines.append(row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table*}")

        return "\n".join(lines)


def _set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flatten_metrics(metrics: Dict) -> Dict[str, float]:
    """Flatten nested metric dicts to a single level."""
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subval in value.items():
                if isinstance(subval, (int, float)):
                    flat[f"{key}/{subkey}"] = float(subval)
        elif isinstance(value, (int, float)):
            flat[key] = float(value)
    return flat


def _aggregate_seeds(seed_results: List[Dict]) -> Dict:
    """Aggregate metrics across seeds."""
    if not seed_results:
        return {}

    all_keys = set()
    for r in seed_results:
        all_keys.update(r.keys())

    aggregated = {}
    for key in all_keys:
        values = [r[key] for r in seed_results if key in r]
        if values and all(isinstance(v, (int, float)) for v in values):
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(v) for v in values],
            }

    return aggregated


def _latex_escape(s: str) -> str:
    replacements = {"_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#"}
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("Benchmarks module ready. Use BenchmarkRunner with trained models and test data.")

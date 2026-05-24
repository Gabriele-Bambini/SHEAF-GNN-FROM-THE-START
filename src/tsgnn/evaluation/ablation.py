"""
Ablation Study Runner for TS-GNN.

Phase 4B: Systematic ablation experiments across 5 axes:
1. Sheaf vs. no sheaf (standard graph Laplacian)
2. FiLM vs. no allele conditioning
3. Stalk dimension d in {2, 4, 8}
4. k-hop radius k in {1, 2, 3}
5. Temporal resolution K in {5, 10, 20}

Generates LaTeX-ready results table with mean +/- std across 3 random seeds.
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AblationRunner:
    """
    Runs systematic ablation experiments for TS-GNN.

    Args:
        base_config: Base configuration dict.
        train_fn: Callable that trains a model given config, returns metrics dict.
        eval_fn: Callable that evaluates a model, returns metrics dict.
        seeds: List of random seeds for repeated experiments.
    """

    def __init__(
        self,
        base_config: dict,
        train_fn,
        eval_fn,
        seeds: Optional[List[int]] = None,
    ):
        self.base_config = base_config
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.seeds = seeds or [42, 123, 456]
        self.results: Dict[str, List[Dict]] = {}

    def run_all(self) -> Dict[str, Dict]:
        """Run all 5 ablation studies."""
        logger.info("Starting ablation studies...")

        studies = {
            "sheaf_vs_no_sheaf": self._ablate_sheaf,
            "film_vs_no_conditioning": self._ablate_film,
            "stalk_dimension": self._ablate_stalk_dim,
            "khop_radius": self._ablate_khop,
            "temporal_resolution": self._ablate_temporal_K,
        }

        all_results = {}
        for name, study_fn in studies.items():
            logger.info(f"Running ablation: {name}")
            all_results[name] = study_fn()

        return all_results

    def _run_with_seeds(self, config: dict, label: str) -> Dict[str, Dict]:
        """Run experiment with multiple seeds and aggregate."""
        seed_results = []
        for seed in self.seeds:
            cfg = copy.deepcopy(config)
            cfg["seed"] = seed
            _set_seed(seed)

            logger.info(f"  {label}, seed={seed}")
            train_metrics = self.train_fn(cfg)
            eval_metrics = self.eval_fn(cfg)
            combined = {**train_metrics, **eval_metrics}
            seed_results.append(combined)

        # Aggregate: mean +/- std
        return _aggregate_results(seed_results)

    def _ablate_sheaf(self) -> Dict:
        """Ablation 1: Sheaf vs. no sheaf (EvolveGCN baseline)."""
        results = {}

        # Full TS-GNN (with sheaf)
        config_sheaf = copy.deepcopy(self.base_config)
        config_sheaf["model"]["type"] = "tsgnn"
        results["with_sheaf"] = self._run_with_seeds(config_sheaf, "TS-GNN (sheaf)")

        # No sheaf (EvolveGCN)
        config_no_sheaf = copy.deepcopy(self.base_config)
        config_no_sheaf["model"]["type"] = "evolvegcn"
        results["without_sheaf"] = self._run_with_seeds(config_no_sheaf, "EvolveGCN (no sheaf)")

        return results

    def _ablate_film(self) -> Dict:
        """Ablation 2: FiLM conditioning vs. no conditioning."""
        results = {}

        # With FiLM
        config_film = copy.deepcopy(self.base_config)
        config_film["model"]["use_allele_conditioning"] = True
        results["with_film"] = self._run_with_seeds(config_film, "TS-GNN (FiLM)")

        # Without FiLM
        config_no_film = copy.deepcopy(self.base_config)
        config_no_film["model"]["use_allele_conditioning"] = False
        results["without_film"] = self._run_with_seeds(config_no_film, "TS-GNN (no FiLM)")

        return results

    def _ablate_stalk_dim(self) -> Dict:
        """Ablation 3: Stalk dimension d in {2, 4, 8} × TP53 alleles (Nature figure)."""
        results = {}
        alleles = ["WT", "R175H", "R273H", "R248W"]
        for d in [2, 4, 8]:
            for allele in alleles:
                config = copy.deepcopy(self.base_config)
                config["model"]["stalk_dim"] = d
                config.setdefault("data", {})["allele"] = allele
                label = f"d={d} {allele}"
                results[label] = self._run_with_seeds(config, label)
        return results

    def _ablate_khop(self) -> Dict:
        """Ablation 4: k-hop radius in {1, 2, 3}."""
        results = {}
        for k in [1, 2, 3]:
            config = copy.deepcopy(self.base_config)
            config["model"]["num_diffusion_steps"] = k
            results[f"k={k}"] = self._run_with_seeds(config, f"k-hop={k}")
        return results

    def _ablate_temporal_K(self) -> Dict:
        """Ablation 5: Temporal resolution K in {5, 10, 20}."""
        results = {}
        for K in [5, 10, 20]:
            config = copy.deepcopy(self.base_config)
            config["data"]["K"] = K
            results[f"K={K}"] = self._run_with_seeds(config, f"K={K}")
        return results

    def generate_latex_table(self, results: Dict[str, Dict]) -> str:
        """Generate LaTeX-ready results table."""
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Ablation study results (mean $\pm$ std across 3 seeds)}")
        lines.append(r"\label{tab:ablation}")

        # Collect all metric names
        all_metrics = set()
        for study_results in results.values():
            for variant_results in study_results.values():
                all_metrics.update(variant_results.keys())
        metric_names = sorted(all_metrics)

        n_cols = len(metric_names) + 1
        lines.append(r"\begin{tabular}{l" + "c" * len(metric_names) + "}")
        lines.append(r"\toprule")

        # Header
        header = "Configuration & " + " & ".join(
            _latex_escape(m) for m in metric_names
        ) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")

        # Rows
        for study_name, study_results in results.items():
            lines.append(r"\multicolumn{" + str(n_cols) + r"}{l}{\textbf{" +
                         _latex_escape(study_name) + r"}} \\")
            for variant, metrics in study_results.items():
                row = _latex_escape(variant) + " & "
                cells = []
                for m in metric_names:
                    if m in metrics:
                        val = metrics[m]
                        if isinstance(val, dict) and "mean" in val:
                            cells.append(f"${val['mean']:.3f} \\pm {val['std']:.3f}$")
                        else:
                            cells.append(f"${val:.3f}$" if isinstance(val, float) else str(val))
                    else:
                        cells.append("--")
                row += " & ".join(cells) + r" \\"
                lines.append(row)
            lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)


def _set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _aggregate_results(seed_results: List[Dict]) -> Dict:
    """Aggregate metrics across seeds: compute mean +/- std."""
    if not seed_results:
        return {}

    all_keys = set()
    for r in seed_results:
        all_keys.update(r.keys())

    aggregated = {}
    for key in all_keys:
        values = [r[key] for r in seed_results if key in r and isinstance(r[key], (int, float))]
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

    return aggregated


def _latex_escape(s: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Demo with mock train/eval functions
    def mock_train(config):
        return {"train_loss": np.random.uniform(0.5, 1.5), "val_loss": np.random.uniform(0.3, 1.0)}

    def mock_eval(config):
        return {"pearson_r": np.random.uniform(0.1, 0.5), "fisher_p": np.random.uniform(0.001, 0.05)}

    base_config = {
        "model": {"type": "tsgnn", "stalk_dim": 4, "use_allele_conditioning": True,
                   "num_diffusion_steps": 3},
        "data": {"K": 10},
    }

    runner = AblationRunner(base_config, mock_train, mock_eval, seeds=[42, 123])
    results = runner.run_all()

    latex = runner.generate_latex_table(results)
    print("LaTeX table generated:")
    print(latex[:500] + "...")

    for study, study_results in results.items():
        print(f"\n{study}:")
        for variant, metrics in study_results.items():
            print(f"  {variant}: {metrics}")

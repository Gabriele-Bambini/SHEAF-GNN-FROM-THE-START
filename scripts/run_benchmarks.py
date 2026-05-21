"""
Run All Baseline Comparisons.

Usage:
    python scripts/run_benchmarks.py --config configs/default.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

logger = logging.getLogger("tsgnn.benchmarks")


def main():
    parser = argparse.ArgumentParser(description="TS-GNN Baseline Benchmarks")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="results/benchmark_table.tex")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    config_path = project_root / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    N = model_cfg.get("input_dim", 50)
    E = model_cfg.get("num_edges", 200)
    d = model_cfg.get("stalk_dim", 4)
    K = config.get("data", {}).get("K", 10)
    esm_dim = model_cfg.get("esm_dim", 1280)

    edge_index = torch.randint(0, N, (2, E))

    # Create all models
    from tsgnn.model.tsgnn import TSGNN
    from tsgnn.model.baselines import EvolveGCN, TemporalGAT, create_tsgnn_no_allele

    models = {
        "tsgnn": TSGNN(
            num_nodes=N, num_edges=E, stalk_dim=d, input_dim=N,
            esm_dim=esm_dim, conditioning_dim=128, edge_index=edge_index,
        ),
        "evolvegcn": EvolveGCN(
            num_nodes=N, input_dim=N, hidden_dim=d * 4, edge_index=edge_index,
        ),
        "tsgnn_no_allele": create_tsgnn_no_allele(
            num_nodes=N, num_edges=E, stalk_dim=d, input_dim=N, edge_index=edge_index,
        ),
        "temporal_gat": TemporalGAT(
            num_nodes=N, input_dim=N, hidden_dim=d * 4, edge_index=edge_index,
        ),
    }

    # Create test data
    from tsgnn.training.trainer import create_synthetic_training_data
    _, test_data, esm_embeddings = create_synthetic_training_data(N=N, E=E, K=K, input_dim=N)

    # Run benchmarks
    from tsgnn.evaluation.benchmarks import BenchmarkRunner

    runner = BenchmarkRunner(
        models=models,
        test_data=test_data,
        esm_embeddings=esm_embeddings,
        stalk_dim=d,
        seeds=[42],
    )

    results = runner.run_all()
    latex = runner.generate_comparison_table(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex)
    logger.info(f"Benchmark results saved to {output_path}")

    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        for metric, val in metrics.items():
            logger.info(f"  {metric}: {val}")


if __name__ == "__main__":
    main()

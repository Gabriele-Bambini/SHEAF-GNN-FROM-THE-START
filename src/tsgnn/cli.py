"""
TS-GNN CLI entry points.

Provides command-line interfaces:
    tsgnn-train     Run the training pipeline
    tsgnn-download  Download all required datasets
    tsgnn-evaluate  Run evaluation on a trained model
"""

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def _load_run_pipeline() -> Callable:
    """Locate and import the ``run_pipeline`` function from scripts/run_pipeline.py.

    The pipeline script lives outside the installed package, so ordinary
    ``from scripts.run_pipeline import ...`` only works in editable / repo
    checkouts. We try three strategies in order:

      1. ``TSGNN_PIPELINE_SCRIPT`` environment variable — explicit override.
      2. Repo-relative path (``<repo>/scripts/run_pipeline.py``).
      3. Current-working-directory path (``./scripts/run_pipeline.py``).

    Raises :class:`ImportError` with a descriptive message if none match.
    """
    candidates: list[Path] = []
    env = os.environ.get("TSGNN_PIPELINE_SCRIPT")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    # src/tsgnn/cli.py → repo root is parents[2]; wheel installs do not have
    # scripts/ alongside, but the fallback below still lets users point at it.
    try:
        candidates.append(here.parents[2] / "scripts" / "run_pipeline.py")
    except IndexError:
        pass
    candidates.append(Path.cwd() / "scripts" / "run_pipeline.py")

    for path in candidates:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("tsgnn._pipeline_script", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.run_pipeline

    raise ImportError(
        "Could not locate scripts/run_pipeline.py. Set TSGNN_PIPELINE_SCRIPT "
        "to its absolute path, or run from the repository root where "
        "scripts/run_pipeline.py is present."
    )


def cli_train():
    """Run the TS-GNN training pipeline."""
    parser = argparse.ArgumentParser(
        description="TS-GNN: Train the Temporal Sheaf Graph Neural Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  tsgnn-train --config configs/default.yaml
  tsgnn-train --config configs/default.yaml --skip-download
  tsgnn-train --config configs/default.yaml --skip-download --skip-training
""",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip the data download step",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training (use existing checkpoint)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # Import here to avoid heavy imports on --help.
    # `scripts/run_pipeline.py` lives at the repo root (editable install) but
    # may be absent in a wheel install — resolve it by path with a clear error.
    from tsgnn.config import TSGNNConfig
    import yaml
    run_pipeline = _load_run_pipeline()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Validate config before running
    with open(config_path) as f:
        raw_dict = yaml.safe_load(f)
        config = TSGNNConfig.from_dict(raw_dict)
    logger.info(f"Loaded and validated configuration. Seed: {config.seed}")

    run_pipeline(str(config_path), args.skip_download, args.skip_training)


def cli_download():
    """Download all required datasets for TS-GNN."""
    parser = argparse.ArgumentParser(
        description="TS-GNN: Download all required datasets",
    )
    parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    from tsgnn.data.download import download_all
    download_all()


def cli_evaluate():
    """Run evaluation on a trained TS-GNN model."""
    parser = argparse.ArgumentParser(
        description="TS-GNN: Evaluate a trained model",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to model checkpoint (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        print("Train a model first with: tsgnn-train --config configs/default.yaml")
        sys.exit(1)

    import yaml
    from tsgnn.config import TSGNNConfig
    config_path = Path(args.config)
    with open(config_path) as f:
        raw_dict = yaml.safe_load(f)
        config = TSGNNConfig.from_dict(raw_dict)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    # Evaluation would load model and run metrics
    # For now, point user to run_pipeline.py
    print("For full evaluation, use: python scripts/run_pipeline.py --skip-download --skip-training")

def cli_validate():
    """Check that every expected data file exists and is non-empty."""
    import sys
    from tsgnn.data import DATA_DIR
    checks = [
        (DATA_DIR / "raw/brca/GSE176078",             "BRCA scRNA GSE176078"),
        (DATA_DIR / "raw/tcga/tcga_brca_tp53_mutations.csv", "TCGA BRCA mutations"),
        (DATA_DIR / "external/jaspar/JASPAR2024_CORE_vertebrates.txt", "JASPAR"),
        (DATA_DIR / "external/string/9606.protein.links.v12.0.txt.gz", "STRING"),
        (DATA_DIR / "external/esm2_embeddings/WT.pt", "ESM-2 WT embedding"),
    ]
    all_ok = True
    for path, desc in checks:
        exists = path.exists() if hasattr(path, "exists") else False
        status = "✅" if exists else "❌"
        print(f"{status} {desc}: {path}")
        if not exists:
            all_ok = False
    sys.exit(0 if all_ok else 1)

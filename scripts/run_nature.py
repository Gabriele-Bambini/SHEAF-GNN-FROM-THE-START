"""
Nature-publication orchestrator.

Drives every falsifiable prediction in NATURE_PLAN.md against the four real
datasets registered in `tsgnn.data.registry`:

    train      = brca_wu2021     (single-cell, primary)
    holdout    = biokey          (single-cell, internal holdout)
    zeroshot   = crc_pelka2021   (single-cell, BRCA→CRC P1)
    bulk       = metabric        (bulk microarray, P3)

Phases (compose by passing --phase ...):
    train      — train TS-GNN on BRCA (compact, CPU-OK)
    holdout    — eval on BIOKEY                                 → results/holdout.json
    zeroshot   — eval on CRC                                    (P1) → results/benchmarks.json
    swap       — allele-swap top-k Jaccard for hotspot pairs     (P2) → results/allele_swap.json
    bulk       — METABRIC enrichment of model-predicted DEGs    (P3) → results/metabric_enrichment.csv
    ablate     — sheaf-off / FiLM-off                            (P4, P5) → results/ablation_table.tex
    all        — all of the above

Smoke mode (--smoke) sub-samples cells / shrinks epochs so the whole pipeline
runs on CPU in a couple of minutes for a sanity check.
"""
from __future__ import annotations

# ── Fix matplotlib/pyparsing conflict on Colab (must run BEFORE any matplotlib import) ──
# Newer pyparsing (>=3.1) changed `ParserElement.resetCache()` so calling it as an
# instance method blows up with "takes 0 positional arguments but 1 was given".
# Pure monkey-patches on pyparsing keep getting bypassed (descriptor resolution),
# so we patch the matplotlib source on disk: wrap the offending call in a try/except.
import os as _os
if _os.name == "nt":
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tsgnn.data import DATA_DIR, load_dataset
from tsgnn.data.nature_adapter import adata_to_tsgnn_tensors, align_to_gene_set
from tsgnn.data.allele import TP53_HOTSPOT_MUTATIONS
from tsgnn.evaluation.allele_swap import expected_effect_curve
from tsgnn.evaluation.bulk_enrichment import enrichment_per_allele
from tsgnn.evaluation.stats import aggregate_seeds
from tsgnn.model.tsgnn import create_tsgnn_from_config
from tsgnn.model.baselines import create_baseline
from sheaf_ts.hpc.manifest import build_runtime_manifest, write_runtime_manifest

logger = logging.getLogger("tsgnn.nature")
RESULTS_ROOT = ROOT / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
ACTIVE_RUN_DIR = RESULTS_ROOT
LATEST_RESULTS_DIR = RESULTS_ROOT / "latest"
PROCESSED_DIR = DATA_DIR / "processed"
ESM_DIR = DATA_DIR / "external" / "esm2_embeddings"


# ─── helpers ─────────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_run_id(config_name: str, phases: List[str], seed: int, smoke: bool) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    config_tag = Path(config_name).stem
    mode = "smoke" if smoke else "full"
    phase_tag = "-".join(phases)
    return f"{stamp}_{config_tag}_seed{seed}_{mode}_{phase_tag}"


def _prepare_result_dirs(
    base_dir: Path,
    run_id: str,
    config_name: str,
    phases: List[str],
    seed: int,
    smoke: bool,
) -> Dict[str, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = base_dir / "runs"
    latest_dir = base_dir / "latest"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "config": config_name,
        "phases": phases,
        "seed": seed,
        "smoke": smoke,
        "run_dir": str(run_dir),
        "latest_dir": str(latest_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    manifest_text = json.dumps(manifest, indent=2)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(manifest_text)
    (base_dir / "latest_run.json").write_text(manifest_text)
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "latest_dir": latest_dir,
        "manifest_path": manifest_path,
    }


def _artifact_targets(name: str) -> List[Path]:
    return [ACTIVE_RUN_DIR / name, LATEST_RESULTS_DIR / name]


def _write_json_artifact(name: str, payload) -> None:
    text = json.dumps(payload, indent=2, default=str)
    for path in _artifact_targets(name):
        path.write_text(text)


def _write_text_artifact(name: str, text: str) -> None:
    for path in _artifact_targets(name):
        path.write_text(text)


def _write_csv_artifact(name: str, df) -> None:
    for path in _artifact_targets(name):
        df.to_csv(path, index=False)


def _strict_science_enabled(cfg: dict, smoke: bool) -> bool:
    env = _os.environ.get("TSGNN_STRICT_SCIENCE")
    if env is not None:
        return env.strip().lower() not in {"0", "false", "no", "off"}
    return bool(cfg.get("science", {}).get("strict_real_data", not smoke))


def _expected_alleles(cfg: dict) -> List[str]:
    return list(dict.fromkeys(["WT", *cfg.get("data", {}).get("alleles", list(TP53_HOTSPOT_MUTATIONS.keys()))]))


def _load_bundle(name: str, cfg: dict, smoke: bool, seed: int, stats: Optional[Dict] = None,
                  pseudotime_method: Optional[str] = None) -> Dict:
    """Load a registry dataset and pack into trainer tensors.

    Caches AnnData as .h5ad after first load so subsequent seeds skip IO.
    Smoke mode subsamples to ≤3000 cells for fast CPU iteration.
    """
    import anndata as _ad
    cache_path = PROCESSED_DIR / f"_cache_{name}.h5ad"
    logger.info("Loading dataset: %s", name)
    if cache_path.exists() and not smoke:
        logger.info("  h5ad cache hit: %s", cache_path)
        adata = _ad.read_h5ad(cache_path)
    else:
        kw = {}
        if name == "biokey":
            kw["malignant_only"] = False
            if smoke:
                kw["sample_n"] = 2000
                kw["random_state"] = seed
        if name == "metabric":
            kw["hotspots_only"] = True
        if name in ("brca", "wu2021") and smoke:
            kw["max_cells"] = 3000
        adata = load_dataset(name, **kw)
        if not smoke:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(cache_path)
            logger.info("  h5ad cached: %s", cache_path)
    if smoke and not name == "metabric":
        n = adata.n_obs
        if n > 3000:
            idx = np.random.default_rng(seed).choice(n, 3000, replace=False)
            adata = adata[idx].copy()
    use_scprint = cfg.get("data", {}).get("use_scprint_grn", False)
    scprint_ckpt = cfg.get("data", {}).get("scprint_ckpt", "jkobject/scPRINT")
    scprint_cache = PROCESSED_DIR
    mu = stats["mu"] if stats else None
    sd = stats["sd"] if stats else None
    genes = stats.get("genes") if stats else None

    bundle = adata_to_tsgnn_tensors(
        adata,
        K=cfg["data"]["K"],
        n_genes=cfg["data"]["n_genes"] if not smoke else 200,
        input_dim=cfg["model"]["input_dim"],
        seed=seed,
        use_scprint_grn=use_scprint and not smoke,
        scprint_cache_dir=scprint_cache,
        scprint_dataset_tag=name,
        scprint_ckpt=scprint_ckpt,
        ref_mu=mu,
        ref_sd=sd,
        ref_genes=genes,
        pseudotime_method=pseudotime_method or cfg.get("data", {}).get("pseudotime_method", "mean_expr"),
    )
    bundle["adata"] = adata
    return bundle


def _build_model(cfg: dict, edge_index: torch.Tensor, n_genes: int, kind: str = "tsgnn"):
    """Instantiate either TS-GNN (via factory) or a registry baseline."""
    cfg = {**cfg}
    cfg.setdefault("model", {})
    cfg["model"] = {**cfg["model"], "input_dim": cfg["model"].get("input_dim", 1)}
    cfg["data"] = {**cfg.get("data", {}), "n_genes": n_genes}
    if kind == "tsgnn":
        cfg["model"]["num_edges"] = edge_index.shape[1]
        return create_tsgnn_from_config(cfg, edge_index)
    # Per-baseline kwargs to satisfy distinct constructor signatures.
    base_kw = dict(
        num_nodes=n_genes,
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"].get("stalk_dim", 4) * 4,
        edge_index=edge_index,
    )
    if kind == "tsgnn_no_allele":
        # create_tsgnn_no_allele has its own signature (no hidden_dim).
        return create_baseline(
            kind,
            num_nodes=n_genes,
            num_edges=edge_index.shape[1],
            stalk_dim=cfg["model"].get("stalk_dim", 4),
            input_dim=cfg["model"]["input_dim"],
            edge_index=edge_index,
            num_diffusion_steps=cfg["model"].get("num_diffusion_steps", 3),
        )
    return create_baseline(kind, **base_kw)


def _train_one(model, bundle, esm, cfg, n_epochs: int, device,
                allele_bundles: Optional[Dict[str, Dict]] = None,
                allele_esm: Optional[Dict[str, torch.Tensor]] = None,
                seed: int = 0):
    """Compact masked-reconstruction trainer reused across all models.

    If ``allele_bundles`` and ``allele_esm`` are provided (multi-allele training),
    each epoch samples a random allele and uses its (X_seq, esm) pair — this
    makes FiLM see diverse allele conditionings and is required for P2/P5 to
    be meaningful.

    Otherwise falls back to single-allele training on ``bundle``/``esm`` (legacy).
    """
    import random as _random
    rng = _random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    multi = allele_bundles is not None and allele_esm is not None and len(allele_bundles) > 1
    if multi:
        alleles = list(allele_bundles.keys())
        # Pre-move each allele bundle's X_seq to device once
        X_by_allele = {a: allele_bundles[a]["X_seq"].to(device) for a in alleles}
        esm_by_allele = {a: allele_esm[a].to(device) for a in alleles}
        # Validation uses the primary bundle (first allele)
        X_val_full = X_by_allele[alleles[0]]
        esm_val = esm_by_allele[alleles[0]]
    else:
        X_val_full = bundle["X_seq"].to(device)
        esm_val = esm
        X_by_allele = None
        esm_by_allele = None

    K = X_val_full.shape[0]
    val_k = max(1, K // 5)
    last_train, last_val = float("nan"), float("nan")
    model.train()
    for epoch in range(n_epochs):
        if multi:
            a = rng.choice(alleles)
            X = X_by_allele[a]
            esm_epoch = esm_by_allele[a]
        else:
            X = X_val_full
            esm_epoch = esm_val
        
        # Temporal Prediction: Predict X[t+1] from X[t]
        # X_train: inputs (0 to K-2)
        # Y_train: targets (1 to K-1)
        X_train = X[:-val_k - 1]
        Y_train = X[1:-val_k]

        opt.zero_grad()
        try:
            out = model(X_train, esm_epoch) if hasattr(model, "forward") else model(X_train)
            preds = out[0] if isinstance(out, tuple) else out
            if isinstance(preds, list):
                preds = torch.stack(preds, dim=0)

            # A3 (predict_delta=True): model returns X(t)+ΔX_pred (absolute).
            # MSE(X(t)+ΔX_pred, X(t+1)) = MSE(ΔX_pred, ΔX_true) — same gradient.
            # Explicit delta loss for baselines that don't have predict_delta
            # (e.g. MLPBaseline): compare deltas directly to avoid identity-cheat.
            use_delta_loss = getattr(model, "predict_delta", False)
            if use_delta_loss:
                # Predictions already include residual; standard MSE is fine.
                loss = torch.mean((preds - Y_train) ** 2)
            else:
                # Non-delta model: compute delta loss explicitly to level the field.
                # delta_pred = preds - X_train;  delta_true = Y_train - X_train
                delta_pred = preds - X_train
                delta_true = Y_train - X_train
                loss = torch.mean((delta_pred - delta_true) ** 2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_train = float(loss.item())
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("[HPC] CUDA OOM during training epoch. Consider reducing K or n_genes.")
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                raise e
            raise e

    last_val = _evaluate_one(model, {"X_seq": X_val_full}, esm_val, device)["val_loss"]
    return {"train_loss": last_train, "val_loss": last_val}


def _evaluate_one(model, bundle, esm, device) -> Dict[str, Optional[float]]:
    """Evaluate forecasting loss without performing optimizer steps."""
    X_full = bundle["X_seq"].to(device)
    if X_full.shape[0] < 2:
        X_in = X_full
        Y_gt = X_full
    else:
        val_k = max(1, X_full.shape[0] // 5)
        X_in = X_full[-val_k - 1 : -1]
        Y_gt = X_full[-val_k:]

    model.eval()
    with torch.no_grad():
        out = model(X_in, esm)
        preds = out[0] if isinstance(out, tuple) else out
        if isinstance(preds, list):
            preds = torch.stack(preds, dim=0)
        preds = preds[-Y_gt.shape[0] :]

        # A3: use delta loss for non-predict_delta models for fair comparison.
        use_delta_loss = getattr(model, "predict_delta", False)
        if use_delta_loss:
            eval_loss = float(((preds - Y_gt) ** 2).mean().item())
        else:
            X_prev = X_in[-Y_gt.shape[0]:]
            delta_pred = preds - X_prev
            delta_true = Y_gt - X_prev
            eval_loss = float(((delta_pred - delta_true) ** 2).mean().item())
    return {"train_loss": None, "val_loss": eval_loss, "eval_loss": eval_loss}


def _predict_t0(model, bundle, esm, device) -> np.ndarray:
    """Return per-gene prediction at t=0 (used for swap & enrichment)."""
    model.eval()
    X = bundle["X_seq"].to(device)
    with torch.no_grad():
        out = model(X, esm)
        preds = out[0] if isinstance(out, tuple) else out
        if isinstance(preds, list):
            preds = preds[0]
        else:
            preds = preds[0]
    return preds.cpu().numpy().squeeze()


def _predict_forecast(model, bundle, esm, device) -> tuple:
    """Forecast the LAST timestep from the preceding ones.

    This is the correct zero-shot target: the model, fed timesteps 0..K-2,
    must predict the future state X[K-1]. Comparing to X[K-1] (unseen during
    this forward pass) is a genuine generalization test — a naive identity
    baseline (e.g. MLP reproducing input) will not score r~1 on this.

    Returns: (pred_last, gt_last) both as 1-D numpy arrays over genes.
    """
    model.eval()
    X_full = bundle["X_seq"].to(device)            # (K, N, D)
    if X_full.shape[0] < 2:
        # Degenerate K=1 bundle — fall back to reconstruction of t=0
        return _predict_t0(model, bundle, esm, device), \
               X_full[0].mean(-1).cpu().numpy()
    X_in = X_full[:-1]                              # feed K-1 steps
    gt_last = X_full[-1].mean(-1).cpu().numpy()     # (N,)  ground-truth last step
    with torch.no_grad():
        out = model(X_in, esm)
        preds = out[0] if isinstance(out, tuple) else out
        if isinstance(preds, list):
            p_last = preds[-1]                       # predicted last step
        else:
            p_last = preds[-1]
    pred_last = p_last.cpu().numpy().squeeze()
    # Collapse feature dim if still present (pred shape (N,) or (N,D))
    if pred_last.ndim > 1:
        pred_last = pred_last.mean(-1)
    return pred_last, gt_last


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten(); b = b.flatten()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _allele_emb(allele: str, esm_dim: int, seed: int, allow_random_fallback: bool) -> torch.Tensor:
    """Deterministic per-allele ESM-2 surrogate (real ESM cache → drop-in here).

    For Phase B with the real cache, swap this for an actual lookup into
    `data/external/esm2_embeddings/`. For the orchestrator smoke / CI run
    we use a per-allele deterministic random vector so swaps are meaningful.
    """
    real = ESM_DIR / f"{allele}.pt"
    if real.exists():
        try:
            return torch.load(real, map_location="cpu").float().reshape(1, esm_dim)
        except Exception:
            if not allow_random_fallback:
                raise RuntimeError(f"Failed to load required ESM embedding: {real}")
    if not allow_random_fallback:
        raise FileNotFoundError(
            f"Missing required ESM embedding for allele '{allele}' at {real}. "
            "Full scientific runs require precomputed embeddings."
        )
    g = torch.Generator().manual_seed(abs(hash(("esm", allele, seed))) % (2**31))
    return torch.randn(1, esm_dim, generator=g)


# ─── phases ──────────────────────────────────────────────────────────────────

def _build_allele_bundles(adata, cfg: dict, reference_genes, seed: int) -> Dict[str, Dict]:
    """Return {allele → bundle} by subsetting AnnData per tp53_allele class.

    Required so FiLM sees diverse allele conditionings during training
    (training on WT-only makes P2/P5 vacuous — model learns to ignore ESM).

    Only produces a per-allele bundle if the allele has enough cells (≥50).
    The gene set and graph are reused from `reference_genes` (BRCA top-variance).
    """
    # Find the allele column
    from tsgnn.data.nature_adapter import _pick_allele_column
    col = _pick_allele_column(adata)
    out: Dict[str, Dict] = {}
    if col is None:
        logger.warning("[multi-allele] no allele column in obs — skipping")
        return out
    vc = adata.obs[col].astype(str).value_counts()
    targets = [a for a in cfg.get("data", {}).get("alleles", []) if a in vc.index and vc[a] >= 50]
    if not targets:
        # Fall back to top-3 most common alleles (any value)
        targets = list(vc.head(3).index)
    logger.info("[multi-allele] building bundles for: %s (cell counts: %s)",
                targets, {a: int(vc.get(a, 0)) for a in targets})

    # Reuse the canonical gene ordering from the aggregate bundle to keep
    # edge_index compatible across alleles (same n_genes, same graph).
    for a in targets:
        mask = adata.obs[col].astype(str) == a
        if int(mask.sum()) < 20:
            continue
        sub = adata[mask].copy()
        b = adata_to_tsgnn_tensors(
            sub,
            K=cfg["data"]["K"],
            n_genes=len(reference_genes),
            input_dim=cfg["model"]["input_dim"],
            seed=seed,
            pseudotime_method=cfg.get("data", {}).get("pseudotime_method", "mean_expr"),
        )
        # Align X_seq to reference gene ordering (same as main bundle)
        aligned = align_to_gene_set(b, reference_genes)
        aligned["allele_label"] = a
        out[a] = aligned
    return out


def phase_train_and_holdout(cfg, smoke, seed) -> Dict:
    dev = _device()
    strict_science = _strict_science_enabled(cfg, smoke)
    train_b = _load_bundle(cfg["datasets"]["train"], cfg, smoke, seed)
    # Fix: Use training-set statistics for holdout normalization (inductive)
    hold_b = _load_bundle(cfg["datasets"]["holdout"], cfg, smoke, seed, stats=train_b["stats"])

    edge_index = train_b["edge_index"].to(dev)
    n_genes = train_b["X_seq"].shape[1]
    model = _build_model(cfg, edge_index, n_genes, kind="tsgnn").to(dev)
    primary_allele = train_b.get("allele_label", "WT")
    esm_primary = _allele_emb(
        primary_allele, cfg["model"]["esm_dim"], seed, allow_random_fallback=not strict_science
    ).to(dev)
    n_ep = 5 if smoke else cfg["training"]["epochs"]

    # Multi-allele training: FiLM must see > 1 allele to learn conditioning.
    allele_bundles = {}
    allele_esm = {}
    if train_b.get("adata") is not None:
        allele_bundles = _build_allele_bundles(
            train_b["adata"], cfg, train_b["gene_names"], seed
        )
        for a, b in allele_bundles.items():
            allele_esm[a] = _allele_emb(
                a, cfg["model"]["esm_dim"], seed, allow_random_fallback=not strict_science
            )
    # Always include the primary bundle under its label so it doesn't get lost
    if primary_allele not in allele_bundles:
        allele_bundles[primary_allele] = train_b
        allele_esm[primary_allele] = esm_primary
    logger.info("[multi-allele] training alleles: %s", list(allele_bundles.keys()))

    train_metrics = _train_one(
        model, train_b, esm_primary, cfg, n_ep, dev,
        allele_bundles=allele_bundles, allele_esm=allele_esm, seed=seed,
    )

    # Project holdout onto the BRCA gene set so we can run the model directly
    hold_aligned = align_to_gene_set(hold_b, train_b["gene_names"])
    eval_metrics = _evaluate_one(model, hold_aligned, esm_primary, dev)
    out = {"train": train_metrics, "holdout": eval_metrics,
           "n_genes": n_genes,
           "allele_train": list(allele_bundles.keys())}
    _write_json_artifact("holdout.json", out)
    logger.info("holdout.json written.")
    return {"model": model, "esm": esm_primary, "train_b": train_b, "hold_b": hold_b,
            "allele_bundles": allele_bundles, "allele_esm": allele_esm,
            "metrics": out}


def _delta_pearson(model, bundle, esm, device) -> tuple:
    """Delta-forecast Pearson: does model predict CHANGE (X[K-1]-X[K-2]) correctly?

    Identity baselines (MLP copy) collapse to r≈0. Real forecast models that
    learn dynamics should score > 0. This is the scientifically defensible
    metric for BRCA→CRC zero-shot because:
      1. pseudotime bins adjacent = highly correlated → absolute-r measures identity
      2. residual r measures "does the model predict the step change"
    """
    model.eval()
    X_full = bundle["X_seq"].to(device)
    if X_full.shape[0] < 2:
        return 0.0, np.zeros(1), np.zeros(1)
    X_in = X_full[:-1]
    gt_last = X_full[-1].mean(-1).cpu().numpy()
    prev_last = X_full[-2].mean(-1).cpu().numpy()
    with torch.no_grad():
        out = model(X_in, esm)
        preds = out[0] if isinstance(out, tuple) else out
        p_last = preds[-1] if isinstance(preds, list) else preds[-1]
    pred_last = p_last.cpu().numpy().squeeze()
    if pred_last.ndim > 1:
        pred_last = pred_last.mean(-1)
    delta_pred = pred_last - prev_last
    delta_gt = gt_last - prev_last
    if delta_pred.std() == 0 or delta_gt.std() == 0:
        return 0.0, delta_pred, delta_gt
    return float(np.corrcoef(delta_pred, delta_gt)[0, 1]), delta_pred, delta_gt


def phase_zeroshot(state, cfg, smoke, seed) -> Dict:
    """P1: BRCA→CRC zero-shot Pearson r vs baselines.

    Reports THREE metrics:
      - crc_pearson_mean: absolute Pearson on X[K-1] (legacy, dominated by identity)
      - crc_delta_pearson: Pearson on (X[K-1]-X[K-2]) (true forecast signal)
      - crc_shuffled_pearson: same as absolute, but on RANDOMLY PERMUTED
        pseudotime bundle. If close to legacy → pseudotime axis is uninformative.
    """
    dev = _device()
    crc_b = _load_bundle(cfg["datasets"]["zeroshot"], cfg, smoke, seed, stats=state["train_b"]["stats"])
    crc_aligned = align_to_gene_set(crc_b, state["train_b"]["gene_names"])

    # Shuffled-pseudotime control: permute the K bins to destroy temporal signal.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(crc_aligned["X_seq"].shape[0])
    crc_shuffled = {
        "X_seq": crc_aligned["X_seq"][perm].clone(),
        "edge_index": crc_aligned["edge_index"],
        "gene_names": crc_aligned["gene_names"],
    }

    # Adjacency correlation — diagnostic
    X_np = crc_aligned["X_seq"].numpy().squeeze(-1)
    adj_corr = float(np.corrcoef(X_np[-2], X_np[-1])[0, 1]) if X_np.shape[0] >= 2 else float("nan")

    results = {"_diagnostic": {"crc_adjacent_bin_corr": adj_corr}}

    def _score(m, esm):
        p_abs, gt_abs = _predict_forecast(m, crc_aligned, esm, dev)
        r_abs = _pearson(p_abs, gt_abs)
        r_delta, _, _ = _delta_pearson(m, crc_aligned, esm, dev)
        p_sh, gt_sh = _predict_forecast(m, crc_shuffled, esm, dev)
        r_shuf = _pearson(p_sh, gt_sh)
        return {
            "crc_pearson_mean": r_abs,
            "crc_delta_pearson": r_delta,
            "crc_shuffled_pearson": r_shuf,
        }

    results["tsgnn"] = _score(state["model"], state["esm"])
    # Baselines
    benchmark_models = list(cfg.get("benchmarks", {}).get("models") or [])
    baseline_models = [name for name in benchmark_models if name != "tsgnn"]
    if not baseline_models:
        raise ValueError("scientific zeroshot requires at least one baseline control")
    for name in baseline_models:
        if name == "tsgnn":
            continue
        edge_index = state["train_b"]["edge_index"].to(dev)
        n_genes = state["train_b"]["X_seq"].shape[1]
        try:
            m = _build_model(cfg, edge_index, n_genes, kind=name).to(dev)
        except Exception as e:
            logger.warning("Skip baseline %s: %s", name, e)
            continue
        _train_one(
            m,
            state["train_b"],
            state["esm"],
            cfg,
            n_epochs=(5 if smoke else cfg["training"]["epochs"]),
            device=dev,
        )
        results[name] = _score(m, state["esm"])
    _write_json_artifact("benchmarks.json", results)
    p1_thresh = cfg["predictions"]["P1_zeroshot_r_threshold"]
    r_abs = results["tsgnn"]["crc_pearson_mean"]
    r_delta = results["tsgnn"]["crc_delta_pearson"]
    logger.info("benchmarks.json: TS-GNN absolute r=%.3f, delta r=%.3f (adj-bin corr=%.3f)",
                r_abs, r_delta, adj_corr)
    logger.info("  P1 threshold %.2f → %s (legacy absolute metric)",
                p1_thresh, "PASS" if r_abs >= p1_thresh else "FAIL")
    return results


def phase_swap(state, cfg, smoke, seed) -> Dict:
    """P2: top-k Jaccard between alleles."""
    dev = _device()
    strict_science = _strict_science_enabled(cfg, smoke)
    out = {}
    for a, b in cfg["allele_swaps"]:
        e_a = _allele_emb(a, cfg["model"]["esm_dim"], seed, allow_random_fallback=not strict_science).to(dev)
        e_b = _allele_emb(b, cfg["model"]["esm_dim"], seed, allow_random_fallback=not strict_science).to(dev)
        # Run on training-set t=0 snapshot so the geometry is in-distribution
        X0 = state["train_b"]["X_seq"][0].to(dev)
        curve = expected_effect_curve(state["model"], X0, e_a, e_b)
        curve["pair"] = f"{a}->{b}"
        out[f"{a}->{b}"] = curve
    _write_json_artifact("allele_swap.json", out)
    logger.info("allele_swap.json written (%d swaps).", len(out))
    return out


def phase_bulk(state, cfg, smoke, seed) -> Dict:
    """P3: METABRIC bulk-enrichment of TS-GNN-predicted DEGs per allele."""
    dev = _device()
    strict_science = _strict_science_enabled(cfg, smoke)
    metabric = load_dataset("metabric", hotspots_only=True)
    # For each hotspot allele present in METABRIC, get the TS-GNN top-k genes
    # under that allele's embedding.
    pred_sets = {}
    for allele in metabric.obs["tp53_allele_canonical"].astype(str).unique():
        if allele in ("WT", "other"):
            continue
        e = _allele_emb(
            allele, cfg["model"]["esm_dim"], seed, allow_random_fallback=not strict_science
        ).to(dev)
        X0 = state["train_b"]["X_seq"][0].to(dev)
        with torch.no_grad():
            out = state["model"](state["train_b"]["X_seq"].to(dev), e)
            preds = out[0] if isinstance(out, tuple) else out
            if isinstance(preds, list):
                preds = preds[0]
            else:
                preds = preds[0]
        scores = preds.cpu().numpy().squeeze()
        topk = np.argsort(-np.abs(scores))[:200]
        pred_sets[allele] = [state["train_b"]["gene_names"][i] for i in topk]
    df = enrichment_per_allele(metabric, pred_sets,
                               fdr=cfg["predictions"]["P3_bh_q_threshold"])
    _write_csv_artifact("metabric_enrichment.csv", df)
    n_pass = int(df["passes_p3"].sum())
    logger.info("metabric_enrichment.csv: %d / %d alleles pass P3 (q < %.2f).",
                n_pass, len(df), cfg["predictions"]["P3_bh_q_threshold"])
    return {"n_pass_p3": n_pass, "n_alleles": len(df)}


def phase_ablate(state, cfg, smoke, seed) -> Dict:
    """P4 (sheaf off) and P5 (FiLM off): single-seed quick check."""
    from tsgnn.evaluation.ablation import AblationRunner
    seeds = [seed] if smoke else cfg.get("benchmarks", {}).get("seeds", [seed])
    strict_science = _strict_science_enabled(cfg, smoke)

    def train_fn(c):
        dev = _device()
        kind = c.get("model", {}).get("type", "tsgnn")
        edge = state["train_b"]["edge_index"].to(dev)
        ng = state["train_b"]["X_seq"].shape[1]
        m = _build_model(c, edge, ng, kind=kind).to(dev)
        esm = _allele_emb(c.get("data", {}).get("allele", "WT"),
                          c["model"]["esm_dim"], seed,
                          allow_random_fallback=not strict_science).to(dev)
        return _train_one(m, state["train_b"], esm, c,
                          n_epochs=(3 if smoke else max(5, c["training"]["epochs"] // 5)),
                          device=dev)

    def eval_fn(c):
        return {}

    runner = AblationRunner(cfg, train_fn, eval_fn, seeds=seeds)
    results = runner.run_all()
    tex = runner.generate_latex_table(results)
    _write_text_artifact("ablation_table.tex", tex)
    logger.info("ablation_table.tex written.")
    return {"studies": list(results.keys())}


# ─── main ────────────────────────────────────────────────────────────────────

PHASES = ["train", "zeroshot", "swap", "bulk", "ablate"]


def main():
    global ACTIVE_RUN_DIR, LATEST_RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nature.yaml")
    ap.add_argument("--phase", nargs="*", default=["all"],
                    help=f"Subset of {PHASES} or 'all'.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    _set_seed(args.seed)
    if _strict_science_enabled(cfg, args.smoke):
        missing = [allele for allele in _expected_alleles(cfg) if not (ESM_DIR / f"{allele}.pt").exists()]
        if missing:
            raise FileNotFoundError(
                "Strict scientific mode is active and required ESM embeddings are missing: "
                f"{missing}. Expected them under {ESM_DIR}."
            )

    todo = PHASES if args.phase == ["all"] else args.phase
    dirs = _prepare_result_dirs(
        RESULTS_ROOT,
        _default_run_id(args.config, todo, args.seed, args.smoke),
        args.config,
        todo,
        args.seed,
        args.smoke,
    )
    ACTIVE_RUN_DIR = dirs["run_dir"]
    LATEST_RESULTS_DIR = dirs["latest_dir"]
    logger.info("Artifacts for this run will be written to %s", ACTIVE_RUN_DIR)
    summary: Dict[str, Dict] = {}
    state = None

    t0 = time.time()
    if "train" in todo or any(p in todo for p in ("zeroshot", "swap", "bulk", "ablate")):
        state = phase_train_and_holdout(cfg, args.smoke, args.seed)
        summary["train"] = state["metrics"]
    if "zeroshot" in todo:
        summary["zeroshot"] = phase_zeroshot(state, cfg, args.smoke, args.seed)
    if "swap" in todo:
        summary["swap"] = phase_swap(state, cfg, args.smoke, args.seed)
    if "bulk" in todo:
        summary["bulk"] = phase_bulk(state, cfg, args.smoke, args.seed)
    if "ablate" in todo:
        summary["ablate"] = phase_ablate(state, cfg, args.smoke, args.seed)

    _write_json_artifact("nature_run_summary.json", summary)
    output_paths = [
        path for path in ACTIVE_RUN_DIR.iterdir()
        if path.is_file() and path.name != "runtime_manifest.json"
    ]
    runtime_manifest = build_runtime_manifest(
        run_id=dirs["run_id"],
        command=[sys.executable, "scripts/run_nature.py", "--config", args.config, "--seed", str(args.seed), "--phase", *todo],
        output_paths=output_paths,
        cwd=ROOT,
        extra={
            "kind": "nature",
            "smoke": bool(args.smoke),
            "phases": todo,
        },
    )
    write_runtime_manifest(runtime_manifest, ACTIVE_RUN_DIR / "runtime_manifest.json")
    write_runtime_manifest(runtime_manifest, LATEST_RESULTS_DIR / "runtime_manifest.json")
    logger.info("Done in %.1fs. Summary saved to %s", time.time() - t0, ACTIVE_RUN_DIR / "nature_run_summary.json")


if __name__ == "__main__":
    main()

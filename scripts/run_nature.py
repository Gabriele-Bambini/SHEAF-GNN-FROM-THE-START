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
import os as _os, importlib.util as _ilu
if _os.name == "nt":
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
def _patch_matplotlib_fontconfig():
    spec = _ilu.find_spec("matplotlib")
    if spec is None or not spec.submodule_search_locations:
        return
    for base in spec.submodule_search_locations:
        fp = _os.path.join(base, "_fontconfig_pattern.py")
        if not _os.path.exists(fp):
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                src = f.read()
            if "# tsgnn-patched" in src:
                return
            # Replace any bare-line `parser.resetCache()` (with leading whitespace) with try/except.
            import re as _re
            new_src, n = _re.subn(
                r"^(\s*)parser\.resetCache\(\)\s*$",
                r"\1try:\n\1    parser.resetCache()\n\1except TypeError:\n\1    pass  # tsgnn-patched: pyparsing>=3.1 incompat",
                src,
                flags=_re.MULTILINE,
            )
            if n:
                with open(fp, "w", encoding="utf-8") as f:
                    f.write(new_src)
                print(f"[run_nature] patched {fp} ({n} site)", flush=True)
        except Exception as _e:
            print(f"[run_nature] could not patch {fp}: {_e}", flush=True)
try:
    _patch_matplotlib_fontconfig()
except Exception as _e:
    print(f"[run_nature] matplotlib patch skipped: {_e}", flush=True)

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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tsgnn.data import DATA_DIR, load_dataset
from tsgnn.data.nature_adapter import adata_to_tsgnn_tensors, align_to_gene_set, intersect_gene_sets
from tsgnn.data.allele import TP53_HOTSPOT_MUTATIONS
from tsgnn.data.allele_classes import allele_label_to_class_id
from tsgnn.evaluation.allele_swap import expected_effect_curve
from tsgnn.evaluation.bulk_enrichment import enrichment_per_allele
from tsgnn.evaluation.geometric_metrics import allele_divergence
from tsgnn.evaluation.stats import aggregate_seeds
from tsgnn.model.tsgnn import TSGNN, create_tsgnn_from_config
from tsgnn.model.baselines import create_baseline

logger = logging.getLogger("tsgnn.nature")
RESULTS_ROOT = ROOT / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
ACTIVE_RUN_DIR = RESULTS_ROOT
LATEST_RESULTS_DIR = RESULTS_ROOT / "latest"
PROCESSED_DIR = DATA_DIR / "processed"
ESM_DIR = DATA_DIR / "external" / "esm2_embeddings"


# ─── helpers ─────────────────────────────────────────────────────────────────


def _allele_separation_loss(model: "TSGNN", margin: float = 0.3) -> torch.Tensor:
    """
    Push SO(4) embeddings of different allele classes apart.

    For each pair (i, j) of different classes, penalize if their SO(4)
    distance < margin. margin=0.3 means classes should be at least 30%
    of max SO(4) distance apart.
    """
    Rs = model.allele_so4.all_rotations()  # (n_classes, d, d)
    n = Rs.shape[0]
    d = Rs.shape[1]
    norm_factor = 2.0 * (d ** 0.5)
    loss = torch.tensor(0.0, device=Rs.device)
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = (Rs[i] - Rs[j]).norm() / norm_factor
            loss = loss + torch.relu(margin - dist)
            n_pairs += 1
    return loss / max(n_pairs, 1)


def _model_forward(model, X: torch.Tensor, allele_label: str, esm, device):
    """
    Dispatch model.forward with the correct allele argument.

    TSGNN (new): expects allele_class LongTensor.
    Baselines: expect esm FloatTensor (or ignore second arg).
    """
    if isinstance(model, TSGNN):
        cls_id = allele_label_to_class_id(allele_label)
        allele_arg = torch.tensor(cls_id, dtype=torch.long, device=device)
    else:
        allele_arg = esm
    return model(X, allele_arg)


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
    return {"run_dir": run_dir, "latest_dir": latest_dir, "manifest_path": manifest_path}


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
    single_allele_label = bundle.get("allele_label", "WT")

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
            a = single_allele_label
            X = X_val_full
            esm_epoch = esm_val
        
        # Temporal Prediction: Predict X[t+1] from X[t]
        # X_train: inputs (0 to K-2)
        # Y_train: targets (1 to K-1)
        X_train = X[:-val_k - 1]
        Y_train = X[1:-val_k]

        opt.zero_grad()
        try:
            out = _model_forward(model, X_train, a, esm_epoch, device)
            preds = out[0] if isinstance(out, tuple) else out
            if isinstance(preds, list):
                preds = torch.stack(preds, dim=0)

            delta_pred = preds - X_train
            delta_true = Y_train - X_train
            recon_loss = torch.mean((delta_pred - delta_true) ** 2)

            # Allele separation: push SO(4) classes apart (TSGNN only)
            if isinstance(model, TSGNN):
                sep_loss = _allele_separation_loss(model, margin=0.3)
                loss = recon_loss + 0.1 * sep_loss
            else:
                sep_loss = torch.tensor(0.0)
                loss = recon_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_train = float(loss.item())

            # Allele divergence diagnostic every 10 epochs
            if isinstance(model, TSGNN) and epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    x_s = X_train[:3]
                    preds_wt = torch.stack(
                        model(x_s, torch.tensor(0, device=device))[0]
                    )
                    preds_struct = torch.stack(
                        model(x_s, torch.tensor(1, device=device))[0]
                    )
                    div = allele_divergence(
                        preds_struct.flatten(), preds_wt.flatten()
                    )
                logger.info(
                    "[epoch %d] allele_divergence(structural vs WT)=%.5f "
                    "recon=%.4f sep=%.4f",
                    epoch, div, recon_loss.item(), sep_loss.item()
                )
                if div < 0.001 and epoch > 20:
                    logger.warning(
                        "allele_divergence < 0.001 after %d epochs -- "
                        "SO4 embedding collapsed. Try increasing allele_init_scale to 0.3.",
                        epoch
                    )
                model.train()
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("[HPC] CUDA OOM during training epoch. Consider reducing K or n_genes.")
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                raise e
            raise e

    last_val = _evaluate_one(model, {"X_seq": X_val_full, "allele_label": single_allele_label},
                             esm_val, device)["val_loss"]
    return {"train_loss": last_train, "val_loss": last_val}


def _evaluate_one(model, bundle, esm, device) -> Dict[str, Optional[float]]:
    """Evaluate forecasting loss without performing optimizer steps."""
    X_full = bundle["X_seq"].to(device)
    allele_label = bundle.get("allele_label", "WT")
    if X_full.shape[0] < 2:
        X_in = X_full
        Y_gt = X_full
    else:
        val_k = max(1, X_full.shape[0] // 5)
        X_in = X_full[-val_k - 1 : -1]
        Y_gt = X_full[-val_k:]

    model.eval()
    with torch.no_grad():
        out = _model_forward(model, X_in, allele_label, esm, device)
        preds = out[0] if isinstance(out, tuple) else out
        if isinstance(preds, list):
            preds = torch.stack(preds, dim=0)
        preds = preds[-Y_gt.shape[0] :]

        X_prev = X_in[-Y_gt.shape[0]:]
        delta_pred = preds - X_prev
        delta_true = Y_gt - X_prev
        eval_loss = float(((delta_pred - delta_true) ** 2).mean().item())
    return {"train_loss": None, "val_loss": eval_loss, "eval_loss": eval_loss}


def _predict_t0(model, bundle, esm, device) -> np.ndarray:
    """Return per-gene prediction at t=0 (used for swap & enrichment)."""
    model.eval()
    X = bundle["X_seq"].to(device)
    allele_label = bundle.get("allele_label", "WT")
    with torch.no_grad():
        out = _model_forward(model, X, allele_label, esm, device)
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
    allele_label = bundle.get("allele_label", "WT")
    if X_full.shape[0] < 2:
        return _predict_t0(model, bundle, esm, device), \
               X_full[0].mean(-1).cpu().numpy()
    X_in = X_full[:-1]
    gt_last = X_full[-1].mean(-1).cpu().numpy()
    with torch.no_grad():
        out = _model_forward(model, X_in, allele_label, esm, device)
        preds = out[0] if isinstance(out, tuple) else out
        p_last = preds[-1] if isinstance(preds, list) else preds[-1]
    pred_last = p_last.cpu().numpy().squeeze()
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
    allele_label = bundle.get("allele_label", "WT")
    if X_full.shape[0] < 2:
        return 0.0, np.zeros(1), np.zeros(1)
    X_in = X_full[:-1]
    gt_last = X_full[-1].mean(-1).cpu().numpy()
    prev_last = X_full[-2].mean(-1).cpu().numpy()
    with torch.no_grad():
        out = _model_forward(model, X_in, allele_label, esm, device)
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

    # Proper gene intersection (fixes P0 blocker: align_to_gene_set zero-fills → r=-0.97)
    train_intersected, crc_aligned, common_genes = intersect_gene_sets(
        state["train_b"], crc_b, min_genes=200, k_neighbors=8
    )
    _write_json_artifact("genes_intersect.json", {
        "n_common": len(common_genes),
        "n_brca_original": len(state["train_b"]["gene_names"]),
        "n_crc_original": len(crc_b["gene_names"]),
        "common_genes": common_genes[:50],  # first 50 for provenance
    })
    logger.info("Gene intersection: %d common genes (BRCA %d, CRC %d)",
                len(common_genes), len(state["train_b"]["gene_names"]), len(crc_b["gene_names"]))

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

    # Retrain TS-GNN on intersected BRCA gene set (same genes as CRC target).
    # Zero-shot transfer: model learns BRCA dynamics on shared genes (BRCA graph),
    # then predicts CRC dynamics on same genes (same graph, CRC expression values).
    # Using BRCA edges for both is standard — tests learned dynamics generalize,
    # not whether graph topology transfers.
    n_common = len(common_genes)
    train_edge = train_intersected["edge_index"].to(dev)

    # Replace CRC edges with BRCA edges (same gene set, model needs consistent topology)
    crc_aligned["edge_index"] = train_intersected["edge_index"]
    crc_shuffled["edge_index"] = train_intersected["edge_index"]

    zs_model = _build_model(cfg, train_edge, n_common, kind="tsgnn").to(dev)
    n_ep = 3 if smoke else cfg["training"]["epochs"]
    _train_one(zs_model, train_intersected, state["esm"], cfg, n_epochs=n_ep, device=dev)

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

    results["tsgnn"] = _score(zs_model, state["esm"])
    # Baselines: also train on intersected BRCA, eval on CRC
    for name in cfg.get("benchmarks", {}).get("models", []):
        if name == "tsgnn":
            continue
        try:
            m = _build_model(cfg, train_edge, n_common, kind=name).to(dev)
        except Exception as e:
            logger.warning("Skip baseline %s: %s", name, e)
            continue
        _train_one(m, train_intersected, state["esm"], cfg,
                   n_epochs=(3 if smoke else max(5, cfg["training"]["epochs"] // 5)),
                   device=dev)
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


def _allele_arg(allele_label: str, model, esm_dim: int, seed: int,
                strict_science: bool, device) -> torch.Tensor:
    """Build the correct allele argument for the given model type."""
    if isinstance(model, TSGNN):
        cls_id = allele_label_to_class_id(allele_label)
        return torch.tensor(cls_id, dtype=torch.long, device=device)
    return _allele_emb(allele_label, esm_dim, seed,
                       allow_random_fallback=not strict_science).to(device)


def phase_swap(state, cfg, smoke, seed) -> Dict:
    """P2: top-k Jaccard between alleles."""
    dev = _device()
    strict_science = _strict_science_enabled(cfg, smoke)
    model = state["model"]
    out = {}
    for a, b in cfg["allele_swaps"]:
        e_a = _allele_arg(a, model, cfg["model"]["esm_dim"], seed, strict_science, dev)
        e_b = _allele_arg(b, model, cfg["model"]["esm_dim"], seed, strict_science, dev)
        X0 = state["train_b"]["X_seq"][0].to(dev)
        curve = expected_effect_curve(model, X0, e_a, e_b)
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
        e = _allele_arg(allele, state["model"], cfg["model"]["esm_dim"],
                        seed, strict_science, dev)
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


def _run_single_seed(cfg, todo, seed, smoke):
    """Run one full pipeline pass for a given seed. Returns summary dict."""
    global ACTIVE_RUN_DIR, LATEST_RESULTS_DIR
    _set_seed(seed)

    dirs = _prepare_result_dirs(
        RESULTS_ROOT,
        _default_run_id("nature.yaml", todo, seed, smoke),
        "configs/nature.yaml",
        todo,
        seed,
        smoke,
    )
    ACTIVE_RUN_DIR = dirs["run_dir"]
    LATEST_RESULTS_DIR = dirs["latest_dir"]
    logger.info("Seed %d: artifacts → %s", seed, ACTIVE_RUN_DIR)

    summary: Dict[str, Dict] = {}
    state = None

    if "train" in todo or any(p in todo for p in ("zeroshot", "swap", "bulk", "ablate")):
        state = phase_train_and_holdout(cfg, smoke, seed)
        summary["train"] = state["metrics"]
    if "zeroshot" in todo:
        summary["zeroshot"] = phase_zeroshot(state, cfg, smoke, seed)
    if "swap" in todo:
        summary["swap"] = phase_swap(state, cfg, smoke, seed)
    if "bulk" in todo:
        summary["bulk"] = phase_bulk(state, cfg, smoke, seed)
    if "ablate" in todo:
        summary["ablate"] = phase_ablate(state, cfg, smoke, seed)

    _write_json_artifact("nature_run_summary.json", summary)
    return summary


def _aggregate_multiseed(all_summaries: list, seeds: list) -> Dict:
    """Aggregate metrics across seeds with mean/std and Wilcoxon test."""
    import numpy as np
    agg: Dict[str, Dict] = {"seeds": seeds, "n_seeds": len(seeds)}

    # Collect scalar metrics from each seed
    metric_arrays: Dict[str, list] = {}
    for s in all_summaries:
        for phase, metrics in s.items():
            if not isinstance(metrics, dict):
                continue
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    key = f"{phase}/{k}"
                    metric_arrays.setdefault(key, []).append(float(v))

    for key, vals in metric_arrays.items():
        arr = np.array(vals)
        agg[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "values": vals,
        }
        # Wilcoxon test vs 0 (is metric significantly non-zero?)
        if len(arr) >= 3:
            try:
                from scipy.stats import wilcoxon
                stat, p = wilcoxon(arr)
                agg[key]["wilcoxon_p"] = float(p)
            except Exception:
                pass

    return agg


def main():
    global ACTIVE_RUN_DIR, LATEST_RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nature.yaml")
    ap.add_argument("--phase", nargs="*", default=["all"],
                    help=f"Subset of {PHASES} or 'all'.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds for multi-seed run (e.g., 42,123,456)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    cfg = yaml.safe_load((ROOT / args.config).read_text())

    if _strict_science_enabled(cfg, args.smoke):
        missing = [allele for allele in _expected_alleles(cfg) if not (ESM_DIR / f"{allele}.pt").exists()]
        if missing:
            raise FileNotFoundError(
                "Strict scientific mode is active and required ESM embeddings are missing: "
                f"{missing}. Expected them under {ESM_DIR}."
            )

    todo = PHASES if args.phase == ["all"] else args.phase

    # Multi-seed mode
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        logger.info("Multi-seed run: %s", seeds)
        t0 = time.time()
        all_summaries = []
        for seed in seeds:
            s = _run_single_seed(cfg, todo, seed, args.smoke)
            all_summaries.append(s)
        agg = _aggregate_multiseed(all_summaries, seeds)
        # Write aggregate
        ACTIVE_RUN_DIR = RESULTS_ROOT / "multiseed_latest"
        ACTIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)
        _write_json_artifact("multiseed_aggregate.json", agg)
        logger.info("Multi-seed done in %.1fs. %d seeds. Aggregate → %s",
                    time.time() - t0, len(seeds), ACTIVE_RUN_DIR)
    else:
        # Single seed (legacy)
        t0 = time.time()
        summary = _run_single_seed(cfg, todo, args.seed, args.smoke)
        logger.info("Done in %.1fs.", time.time() - t0)


if __name__ == "__main__":
    main()

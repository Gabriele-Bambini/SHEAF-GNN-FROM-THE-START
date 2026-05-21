"""Ovarian/HGSOC scRNA-seq loader for BRCA transfer tests.

Dataset: GSE184880, processed GEO supplementary MTX/TSV files.
We intentionally keep only `Cancer*` samples for transfer evaluation to avoid
the normal-vs-tumor gradient issue found in CRC.
"""
from __future__ import annotations

import gzip
import logging
from pathlib import Path
import re
import tarfile
from typing import Optional

import numpy as np
import pandas as pd

from tsgnn.data import RAW_DIR

logger = logging.getLogger(__name__)

OVARIAN_ACCESSION = "GSE184880"
OVARIAN_RAW_DIR = RAW_DIR / "ovarian_hgsoc_gse184880"
OVARIAN_RAW_TAR = OVARIAN_RAW_DIR / "GSE184880_RAW.tar"
OVARIAN_RAW_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE184nnn/GSE184880/suppl/GSE184880_RAW.tar"

GSE158722_ACCESSION = "GSE158722"
GSE158722_RAW_DIR = RAW_DIR / "ovarian_hgsoc_gse158722"
GSE158722_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE158nnn/GSE158722/suppl"
GSE158722_ANNOTATION_FILE = "GSE158722.cell_annotations.txt.gz"
GSE158722_PATIENT_IDS = tuple([f"P{i:02d}" for i in range(1, 22)] + ["P23", "P24"])

GSE154600_ACCESSION = "GSE154600"
GSE154600_RAW_DIR = RAW_DIR / "ovarian_hgsoc_gse154600"
GSE154600_RAW_TAR = GSE154600_RAW_DIR / "GSE154600_RAW.tar"
GSE154600_RAW_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154600/suppl/GSE154600_RAW.tar"

GSE146026_ACCESSION = "GSE146026"
GSE146026_RAW_DIR = RAW_DIR / "ovarian_hgsoc_gse146026"
GSE146026_LOG_FILE = "GSE146026_Izar_HGSOC_ascites_10x_log.tsv.gz"
GSE146026_LOG_URL = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE146nnn/GSE146026/suppl/{GSE146026_LOG_FILE}"
GSE146026_ANNOTATION_GLOBS = (
    "GSE146026*annotation*.tsv*",
    "GSE146026*annotation*.csv*",
    "GSE146026*annot*.tsv*",
    "GSE146026*annot*.csv*",
    "GSE146026*metadata*.tsv*",
    "GSE146026*metadata*.csv*",
)
GSE146026_CELL_ID_COLUMNS = ("cell_id", "Cell", "Cell_ID", "barcode", "10x_barcode", "cell", "NAME")
GSE146026_CELL_TYPE_COLUMNS = (
    "cell_type",
    "CellType",
    "celltype",
    "annotation",
    "cluster_annotation",
    "SingleR_Encode_main_type",
)
GSE146026_MALIGNANCY_COLUMNS = (
    "normal_or_cancer",
    "Normal.or.Cancer",
    "malignant",
    "malignancy",
    "is_malignant",
    "tumor_type",
    "tumour_type",
    "cancer_status",
)


def ovarian_available() -> bool:
    return any(OVARIAN_RAW_DIR.glob("*_Cancer*.matrix.mtx.gz"))


def ovarian_hgsoc_gse158722_available() -> bool:
    return (GSE158722_RAW_DIR / GSE158722_ANNOTATION_FILE).exists() and any(
        GSE158722_RAW_DIR.glob("GSE158722_P*.counts.txt.gz")
    )


def ovarian_hgsoc_gse154600_available() -> bool:
    return any(GSE154600_RAW_DIR.glob("*_matrix.mtx.gz"))


def ovarian_hgsoc_gse146026_available() -> bool:
    return (GSE146026_RAW_DIR / GSE146026_LOG_FILE).exists()


def download_ovarian_scrna() -> Path:
    """Download/extract GSE184880 processed MTX files from GEO."""
    import requests

    OVARIAN_RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not OVARIAN_RAW_TAR.exists():
        tmp = OVARIAN_RAW_TAR.with_suffix(".tar.tmp")
        logger.info("Downloading %s to %s", OVARIAN_RAW_URL, OVARIAN_RAW_TAR)
        with requests.get(OVARIAN_RAW_URL, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        tmp.replace(OVARIAN_RAW_TAR)
    if not ovarian_available():
        with tarfile.open(OVARIAN_RAW_TAR, "r") as archive:
            archive.extractall(OVARIAN_RAW_DIR)
    return OVARIAN_RAW_DIR


def _download_url(url: str, out: Path) -> None:
    import requests

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return
    tmp = out.with_suffix(out.suffix + ".tmp")
    logger.info("Downloading %s to %s", url, out)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(out)


def download_ovarian_hgsoc_gse158722() -> Path:
    """Download GSE158722 processed counts and cell annotations from GEO."""
    GSE158722_RAW_DIR.mkdir(parents=True, exist_ok=True)
    _download_url(
        f"{GSE158722_BASE_URL}/{GSE158722_ANNOTATION_FILE}",
        GSE158722_RAW_DIR / GSE158722_ANNOTATION_FILE,
    )
    for patient_id in GSE158722_PATIENT_IDS:
        filename = f"GSE158722_{patient_id}.counts.txt.gz"
        _download_url(f"{GSE158722_BASE_URL}/{filename}", GSE158722_RAW_DIR / filename)
    return GSE158722_RAW_DIR


def download_ovarian_hgsoc_gse154600() -> Path:
    """Download/extract GSE154600 processed 10X MTX files from GEO."""
    import requests
    from requests import exceptions as req_exc

    GSE154600_RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not GSE154600_RAW_TAR.exists():
        tmp = GSE154600_RAW_TAR.with_suffix(".tar.tmp")
        logger.info("Downloading %s to %s", GSE154600_RAW_URL, GSE154600_RAW_TAR)
        try:
            with requests.get(GSE154600_RAW_URL, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
        except req_exc.RequestException as exc:  # pragma: no cover - exercised via mocks in tests
            raise RuntimeError(
                "Failed to download GSE154600 from GEO (network/proxy blocked or unavailable). "
                f"Expected tarball path: {GSE154600_RAW_TAR}. "
                "Fix: pre-download `GSE154600_RAW.tar` to that path (or set `TSGNN_DATA_DIR` to a data cache with it), "
                "then re-run. Offline/sandbox runs must use a pre-populated cache."
            ) from exc
        tmp.replace(GSE154600_RAW_TAR)
    if not ovarian_hgsoc_gse154600_available():
        with tarfile.open(GSE154600_RAW_TAR, "r") as archive:
            archive.extractall(GSE154600_RAW_DIR)
    return GSE154600_RAW_DIR


def download_ovarian_hgsoc_gse146026() -> Path:
    """Download GSE146026 processed 10x log expression matrix from GEO."""
    GSE146026_RAW_DIR.mkdir(parents=True, exist_ok=True)
    _download_url(GSE146026_LOG_URL, GSE146026_RAW_DIR / GSE146026_LOG_FILE)
    return GSE146026_RAW_DIR


def _matrix_prefix(path: Path) -> str:
    for suffix in (".matrix.mtx.gz", "_matrix.mtx.gz"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.name.replace(".mtx.gz", "")


def _sample_name(path: Path) -> str:
    name = _matrix_prefix(path)
    match = re.match(r"^GSM\d+_(.+)$", name)
    return match.group(1) if match else name


def _read_genes(path: Path) -> list[str]:
    genes: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.rstrip("\n").split("\t")
            if not parts:
                continue
            genes.append(parts[1] if len(parts) > 1 and parts[1] else parts[0])
    return _make_unique_gene_names(genes)


def _make_unique_gene_names(raw_genes: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    genes: list[str] = []
    for gene in raw_genes:
        if gene not in seen:
            seen[gene] = 0
            genes.append(gene)
        else:
            seen[gene] += 1
            genes.append(f"{gene}-{seen[gene]}")
    return genes


def _metadata_row(frame: pd.DataFrame, row_name: str, default: str) -> pd.Series:
    if row_name not in frame.index:
        return pd.Series(default, index=frame.columns, dtype=object)
    return frame.loc[row_name].astype(str)


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        hit = lookup.get(candidate.lower())
        if hit is not None:
            return hit
    return None


def _read_annotation_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    sep = "," if ".csv" in suffixes else "\t"
    return pd.read_csv(path, sep=sep, compression="infer", low_memory=False)


def _normal_cancer_label(value: object) -> str:
    raw = str(value).strip().lower()
    if raw in {"", "nan", "none", "unknown"}:
        return "Unknown"
    if raw in {"1", "true", "yes", "y", "cancer", "malignant", "tumor", "tumour", "neoplastic"}:
        return "Cancer"
    if raw in {"0", "false", "no", "n", "normal", "benign", "nonmalignant", "non-malignant", "non malignant"}:
        return "Normal"
    if "non-malign" in raw or "nonmalign" in raw:
        return "Normal"
    if "malignant" in raw or "tumor" in raw or "tumour" in raw or "cancer" in raw:
        return "Cancer"
    if "normal" in raw or "immune" in raw or "stromal" in raw or "fibroblast" in raw or "endothelial" in raw:
        return "Normal"
    return "Unknown"


def _find_gse146026_annotation_file(raw_dir: Path) -> Path | None:
    for pattern in GSE146026_ANNOTATION_GLOBS:
        for candidate in sorted(raw_dir.glob(pattern)):
            if candidate.name == GSE146026_LOG_FILE or candidate.name.endswith(".tmp"):
                continue
            return candidate
    return None


def _apply_gse146026_annotation_overlay(
    obs: pd.DataFrame,
    *,
    annotation_file: Path,
    min_annotation_fraction: float,
) -> pd.DataFrame:
    annotation = _read_annotation_table(annotation_file)
    cell_col = _first_existing_column(annotation, GSE146026_CELL_ID_COLUMNS)
    if cell_col is None:
        raise ValueError(f"GSE146026 annotation {annotation_file} lacks a cell/barcode column")
    malignancy_col = _first_existing_column(annotation, GSE146026_MALIGNANCY_COLUMNS)
    if malignancy_col is None:
        raise ValueError(f"GSE146026 annotation {annotation_file} lacks direct malignant/normal labels")

    cell_type_col = _first_existing_column(annotation, GSE146026_CELL_TYPE_COLUMNS)
    keyed = annotation.copy()
    keyed["_cell_key"] = keyed[cell_col].astype(str)
    keyed = keyed.drop_duplicates("_cell_key", keep="first").set_index("_cell_key", drop=False)

    matched_rows: list[pd.Series | None] = []
    for _, row in obs.iterrows():
        keys = [str(row.name), str(row.get("matrix_cell_id", "")), str(row.get("barcode", ""))]
        hit = next((keyed.loc[key] for key in keys if key and key in keyed.index), None)
        matched_rows.append(hit)

    matched = [row for row in matched_rows if row is not None]
    coverage = len(matched) / len(obs) if len(obs) else 0.0
    if coverage < min_annotation_fraction:
        raise ValueError(
            f"GSE146026 annotation coverage {coverage:.3f} below required {min_annotation_fraction:.3f}"
        )

    normal_or_cancer = []
    cell_types = []
    for row, existing_cell_type in zip(matched_rows, obs["cell_type"].astype(str).tolist(), strict=True):
        if row is None:
            normal_or_cancer.append("Unknown")
            cell_types.append(existing_cell_type)
            continue
        normal_or_cancer.append(_normal_cancer_label(row[malignancy_col]))
        cell_types.append(str(row[cell_type_col]) if cell_type_col is not None else existing_cell_type)

    out = obs.copy()
    out["normal_or_cancer"] = normal_or_cancer
    out["cell_type"] = cell_types
    out["specimen_type"] = "ascites_annotated"
    out["is_cancer_sample"] = out["normal_or_cancer"].eq("Cancer")
    out["annotation_source"] = annotation_file.name
    out["annotation_coverage_fraction"] = coverage
    if not out["normal_or_cancer"].isin(["Cancer", "Normal"]).any():
        raise ValueError(f"GSE146026 annotation {annotation_file} has no usable Cancer/Normal labels")
    return out


def _read_barcodes(path: Path, *, sample_id: str) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [f"{sample_id}:{line.strip()}" for line in handle if line.strip()]


def _companion_path(matrix_path: Path, prefix: str, stem: str) -> Path:
    candidates = [
        matrix_path.with_name(f"{prefix}.{stem}.tsv.gz"),
        matrix_path.with_name(f"{prefix}_{stem}.tsv.gz"),
    ]
    if stem == "genes":
        candidates.extend(
            [
                matrix_path.with_name(f"{prefix}.features.tsv.gz"),
                matrix_path.with_name(f"{prefix}_features.tsv.gz"),
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_one_sample(matrix_path: Path, *, force_cancer: Optional[bool] = None):
    import anndata
    import scipy.io

    sample_id = _sample_name(matrix_path)
    prefix = _matrix_prefix(matrix_path)
    genes_path = _companion_path(matrix_path, prefix, "genes")
    barcodes_path = _companion_path(matrix_path, prefix, "barcodes")
    if not genes_path.exists() or not barcodes_path.exists():
        raise FileNotFoundError(f"missing genes/barcodes for {matrix_path.name}")

    matrix = scipy.io.mmread(str(matrix_path)).tocsr().T
    genes = _read_genes(genes_path)
    cells = _read_barcodes(barcodes_path, sample_id=sample_id)
    is_cancer_sample = bool(force_cancer) if force_cancer is not None else "cancer" in sample_id.lower()
    normal_or_cancer = "Cancer" if is_cancer_sample else "Normal"
    obs = pd.DataFrame(
        {
            "sample_id": sample_id,
            "platform": "10x_mtx_geo",
            "timepoint": "unspecified",
            "normal_or_cancer": normal_or_cancer,
            "cell_type": "sample_level_unlabeled",
            "specimen_type": "cancer" if is_cancer_sample else "normal",
            "is_cancer_sample": is_cancer_sample,
            "tp53_allele": "unknown",
        },
        index=pd.Index(cells, name="cell_id"),
    )
    return anndata.AnnData(X=matrix.astype(np.float32), obs=obs, var=pd.DataFrame(index=genes))


def load_ovarian_data(
    sample_n: Optional[int] = 20000,
    random_state: int = 42,
    cancer_only: bool = True,
):
    """Load GSE184880 ovarian/HGSOC scRNA-seq as AnnData."""
    try:
        import anndata
    except ImportError as exc:
        raise ImportError("ovarian loader requires anndata") from exc

    if not any(OVARIAN_RAW_DIR.glob("*.matrix.mtx.gz")):
        download_ovarian_scrna()
    matrix_files = sorted(OVARIAN_RAW_DIR.glob("*.matrix.mtx.gz"))
    if cancer_only:
        matrix_files = [path for path in matrix_files if "Cancer" in path.name]
    if not matrix_files:
        raise ValueError("no GSE184880 cancer samples available")

    samples = [_load_one_sample(path) for path in matrix_files]
    adata = anndata.concat(samples, join="outer", merge="same", label=None)
    adata.var_names_make_unique()
    if sample_n is not None and sample_n < adata.n_obs:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(adata.n_obs, size=sample_n, replace=False)
        adata = adata[idx].copy()
    logger.info("Loaded ovarian GSE184880: %s cells x %s genes", adata.n_obs, adata.n_vars)
    return adata


def _gse158722_patient_id(path: Path) -> str:
    match = re.match(r"^GSE158722_(P\d+)\.counts\.txt\.gz$", path.name)
    return match.group(1) if match else path.name.split(".")[0]


def _load_gse158722_counts(path: Path, annotations: pd.DataFrame):
    import anndata
    import scipy.sparse as sp

    patient_id = _gse158722_patient_id(path)
    patient_cells = annotations.index[annotations["Sample.name"].astype(str) == patient_id].tolist()
    if not patient_cells:
        return None

    header = pd.read_csv(path, sep="\t", compression="gzip", nrows=0).columns.tolist()
    cell_cols = [cell for cell in patient_cells if cell in header]
    if not cell_cols:
        return None

    usecols = ["Gene Symbol", "Gene ID"] + cell_cols
    counts = pd.read_csv(path, sep="\t", compression="gzip", usecols=usecols, low_memory=False)
    genes = _make_unique_gene_names(counts["Gene Symbol"].astype(str).tolist())
    matrix = sp.csr_matrix(counts[cell_cols].to_numpy(dtype=np.float32).T)
    meta = annotations.loc[cell_cols]
    obs = pd.DataFrame(
        {
            "platform": meta["Platform"].astype(str).to_numpy(),
            "sample_id": meta["Sample.name"].astype(str).to_numpy(),
            "timepoint": meta["time"].astype(str).to_numpy(),
            "cluster": meta["Cluster"].astype(str).to_numpy(),
            "normal_or_cancer": meta["Normal.or.Cancer"].astype(str).to_numpy(),
            "cell_type": meta["SingleR_Encode_main_type"].astype(str).to_numpy(),
            "specimen_type": "malignant_fluid",
            "is_cancer_sample": meta["Normal.or.Cancer"].astype(str).str.lower().eq("cancer").to_numpy(),
            "tp53_allele": "unknown",
        },
        index=pd.Index(cell_cols, name="cell_id"),
    )
    return anndata.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=genes))


def load_ovarian_hgsoc_gse158722_data(
    sample_n: Optional[int] = 20000,
    random_state: int = 42,
    malignant_only: bool = True,
):
    """Load annotated malignant-fluid HGSOC scRNA-seq from GSE158722."""
    try:
        import anndata
    except ImportError as exc:
        raise ImportError("GSE158722 ovarian loader requires anndata") from exc

    if not ovarian_hgsoc_gse158722_available():
        download_ovarian_hgsoc_gse158722()

    annotation_path = GSE158722_RAW_DIR / GSE158722_ANNOTATION_FILE
    annotations = pd.read_csv(annotation_path, sep="\t", compression="gzip")
    if "Cell" not in annotations:
        raise ValueError("GSE158722 annotation file is missing Cell column")
    annotations = annotations.set_index("Cell", drop=False)
    if malignant_only:
        annotations = annotations[annotations["Normal.or.Cancer"].astype(str).str.lower().eq("cancer")]
    if annotations.empty:
        raise ValueError("no annotated GSE158722 cancer cells available")

    if sample_n is not None and sample_n < len(annotations):
        annotations = annotations.sample(n=sample_n, random_state=random_state)

    samples = []
    for counts_path in sorted(GSE158722_RAW_DIR.glob("GSE158722_P*.counts.txt.gz")):
        sample = _load_gse158722_counts(counts_path, annotations)
        if sample is not None and sample.n_obs > 0:
            samples.append(sample)
    if not samples:
        raise ValueError("no GSE158722 count matrices overlapped annotated cancer cells")

    adata = anndata.concat(samples, join="outer", merge="same", label=None)
    adata.var_names_make_unique()
    logger.info("Loaded ovarian GSE158722: %s cells x %s genes", adata.n_obs, adata.n_vars)
    return adata


def load_ovarian_hgsoc_gse154600_data(
    sample_n: Optional[int] = 20000,
    random_state: int = 42,
):
    """Load GSE154600 primary HGSOC tumor 10X scRNA-seq as AnnData."""
    try:
        import anndata
    except ImportError as exc:
        raise ImportError("GSE154600 ovarian loader requires anndata") from exc

    if not ovarian_hgsoc_gse154600_available():
        download_ovarian_hgsoc_gse154600()
    matrix_files = sorted(GSE154600_RAW_DIR.glob("*_matrix.mtx.gz"))
    if not matrix_files:
        raise ValueError("no GSE154600 tumor samples available")

    samples = [_load_one_sample(path, force_cancer=True) for path in matrix_files]
    adata = anndata.concat(samples, join="outer", merge="same", label=None)
    adata.var_names_make_unique()
    if sample_n is not None and sample_n < adata.n_obs:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(adata.n_obs, size=sample_n, replace=False)
        adata = adata[idx].copy()
    logger.info("Loaded ovarian GSE154600: %s cells x %s genes", adata.n_obs, adata.n_vars)
    return adata


def load_ovarian_hgsoc_gse146026_data(
    sample_n: Optional[int] = 20000,
    random_state: int = 42,
    allow_unlabeled_ascites: bool = False,
    annotation_file: Optional[Path] = None,
    malignant_only: bool = True,
    min_annotation_fraction: float = 0.95,
):
    """Load GSE146026 processed HGSOC ascites matrix for audit/stress tests only."""
    try:
        import anndata
        import scipy.sparse as sp
    except ImportError as exc:
        raise ImportError("GSE146026 ovarian loader requires anndata and scipy") from exc

    if not ovarian_hgsoc_gse146026_available():
        download_ovarian_hgsoc_gse146026()

    path = GSE146026_RAW_DIR / GSE146026_LOG_FILE
    frame = pd.read_csv(path, sep="\t", compression="gzip", index_col=0, low_memory=False)
    metadata_rows = ["10x_barcode", "patient", "time", "sample_ID", "clst", "TSNE_x", "TSNE_y"]
    expr = frame.drop(index=[row for row in metadata_rows if row in frame.index])
    if expr.empty:
        raise ValueError("GSE146026 processed matrix has no gene expression rows")

    barcodes = _metadata_row(frame, "10x_barcode", "").replace("", pd.NA).fillna(pd.Series(frame.columns, index=frame.columns))
    clusters = _metadata_row(frame, "clst", "unknown")
    obs = pd.DataFrame(
        {
            "matrix_cell_id": frame.columns.astype(str).to_numpy(),
            "barcode": barcodes.astype(str).to_numpy(),
            "sample_id": _metadata_row(frame, "sample_ID", "unknown").to_numpy(),
            "patient_id": _metadata_row(frame, "patient", "unknown").to_numpy(),
            "timepoint": _metadata_row(frame, "time", "unknown").to_numpy(),
            "cluster": clusters.to_numpy(),
            "platform": "10x_processed_log_geo",
            "normal_or_cancer": "Unlabeled",
            "cell_type": [f"unlabeled_ascites_cluster_{value}" for value in clusters.to_numpy()],
            "specimen_type": "ascites_unlabeled",
            "is_cancer_sample": False,
            "tp53_allele": "unknown",
        },
        index=pd.Index(barcodes.astype(str).to_numpy(), name="cell_id"),
    )
    resolved_annotation = Path(annotation_file) if annotation_file is not None else _find_gse146026_annotation_file(GSE146026_RAW_DIR)
    if resolved_annotation is not None:
        obs = _apply_gse146026_annotation_overlay(
            obs,
            annotation_file=resolved_annotation,
            min_annotation_fraction=min_annotation_fraction,
        )
    elif not allow_unlabeled_ascites:
        raise ValueError(
            "GSE146026 lacks direct malignant labels in the processed GEO matrix; "
            "provide a verified annotation_file or pass allow_unlabeled_ascites=True only for audit/stress tests"
        )

    matrix = sp.csr_matrix(expr.to_numpy(dtype=np.float32).T)
    adata = anndata.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=_make_unique_gene_names(expr.index.astype(str).tolist())))
    adata.var_names_make_unique()
    if resolved_annotation is not None and malignant_only:
        adata = adata[adata.obs["is_cancer_sample"].astype(bool).to_numpy()].copy()
        if adata.n_obs == 0:
            raise ValueError(f"GSE146026 annotation {resolved_annotation} has no malignant/cancer cells")
    if sample_n is not None and sample_n < adata.n_obs:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(adata.n_obs, size=sample_n, replace=False)
        adata = adata[idx].copy()
    logger.info("Loaded ovarian GSE146026 audit matrix: %s cells x %s genes", adata.n_obs, adata.n_vars)
    return adata


__all__ = [
    "OVARIAN_ACCESSION",
    "OVARIAN_RAW_DIR",
    "OVARIAN_RAW_TAR",
    "OVARIAN_RAW_URL",
    "GSE158722_ACCESSION",
    "GSE158722_RAW_DIR",
    "GSE158722_BASE_URL",
    "GSE146026_ACCESSION",
    "GSE146026_RAW_DIR",
    "GSE146026_LOG_URL",
    "download_ovarian_scrna",
    "download_ovarian_hgsoc_gse158722",
    "download_ovarian_hgsoc_gse146026",
    "load_ovarian_data",
    "load_ovarian_hgsoc_gse158722_data",
    "load_ovarian_hgsoc_gse146026_data",
    "ovarian_available",
    "ovarian_hgsoc_gse158722_available",
    "ovarian_hgsoc_gse146026_available",
]

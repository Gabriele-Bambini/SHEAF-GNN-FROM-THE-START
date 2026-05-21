"""Patient-disjoint split manifests for non-transductive evaluation."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import random
from typing import Mapping, Sequence

__all__ = ["make_patient_split_manifest", "validate_patient_split_manifest"]


def _stable_hash(items: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for item in sorted(items):
        digest.update(item.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def make_patient_split_manifest(
    records: Sequence[Mapping[str, object]],
    *,
    patient_key: str = "patient_id",
    label_key: str = "tp53_label",
    label_provenance: str,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict:
    """Create a deterministic patient-disjoint split manifest.

    The function is intentionally strict: one patient cannot carry conflicting
    labels, and labels must have explicit provenance.
    """
    if not label_provenance:
        raise ValueError("label_provenance is required")
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("split fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    labels_by_patient: dict[str, set[str]] = defaultdict(set)
    for record in records:
        patient = str(record.get(patient_key, "")).strip()
        label = str(record.get(label_key, "")).strip()
        if not patient:
            raise ValueError(f"record missing {patient_key}")
        if not label:
            raise ValueError(f"record missing {label_key} for patient {patient}")
        labels_by_patient[patient].add(label)

    conflicts = {
        patient: sorted(labels)
        for patient, labels in labels_by_patient.items()
        if len(labels) > 1
    }
    if conflicts:
        raise ValueError(f"conflicting labels for patients: {conflicts}")

    patients = sorted(labels_by_patient)
    rng = random.Random(seed)
    shuffled = patients[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    if n >= 3:
        n_test = max(1, n_test)
        n_val = max(1, n_val)
    if n_test + n_val >= n and n > 0:
        n_test = max(0, min(n_test, n - 2))
        n_val = max(0, min(n_val, n - n_test - 1))

    test = sorted(shuffled[:n_test])
    val = sorted(shuffled[n_test:n_test + n_val])
    train = sorted(shuffled[n_test + n_val:])
    manifest = {
        "split_policy": "patient_disjoint",
        "seed": int(seed),
        "patient_key": patient_key,
        "label_key": label_key,
        "label_provenance": label_provenance,
        "num_patients": n,
        "patient_label_sha256": _stable_hash(
            [f"{patient}:{next(iter(labels_by_patient[patient]))}" for patient in patients]
        ),
        "splits": {
            "train": train,
            "val": val,
            "test": test,
        },
    }
    validate_patient_split_manifest(manifest)
    return manifest


def validate_patient_split_manifest(manifest: Mapping[str, object]) -> None:
    if manifest.get("split_policy") != "patient_disjoint":
        raise ValueError("split_policy must be patient_disjoint")
    if not manifest.get("label_provenance"):
        raise ValueError("label_provenance is required")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("splits mapping is required")

    split_sets = {}
    for name in ("train", "val", "test"):
        values = splits.get(name)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"splits.{name} must be a sequence")
        split_sets[name] = {str(value) for value in values}

    overlap_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in overlap_pairs:
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise ValueError(f"patient overlap between {left} and {right}: {sorted(overlap)}")

    observed = sum(len(values) for values in split_sets.values())
    expected = int(manifest.get("num_patients", -1))
    if expected != observed:
        raise ValueError(f"num_patients={expected} does not match split size {observed}")

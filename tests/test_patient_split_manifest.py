import pytest

from sheaf_ts.data.split_manifest import (
    make_patient_split_manifest,
    validate_patient_split_manifest,
)


def test_patient_split_manifest_rejects_conflicting_patient_labels():
    records = [
        {"patient_id": "P1", "tp53_label": "WT"},
        {"patient_id": "P1", "tp53_label": "R175H"},
    ]

    with pytest.raises(ValueError, match="conflicting labels"):
        make_patient_split_manifest(records, label_provenance="matched_wes")


def test_patient_split_manifest_is_patient_disjoint_and_auditable():
    records = [
        {"patient_id": f"P{i}", "tp53_label": "WT" if i % 2 else "R175H"}
        for i in range(12)
    ]

    manifest = make_patient_split_manifest(
        records,
        label_provenance="matched_wes",
        val_fraction=0.25,
        test_fraction=0.25,
        seed=11,
    )
    validate_patient_split_manifest(manifest)

    train = set(manifest["splits"]["train"])
    val = set(manifest["splits"]["val"])
    test = set(manifest["splits"]["test"])
    assert train
    assert val
    assert test
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert manifest["label_provenance"] == "matched_wes"
    assert manifest["num_patients"] == 12

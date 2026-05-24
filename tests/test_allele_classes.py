import pytest
from tsgnn.data.allele_classes import (
    ALLELE_TO_CLASS,
    CLASS_NAMES,
    N_ALLELE_CLASSES,
    allele_label_to_class_id,
    get_class_name,
)


def test_n_classes():
    assert N_ALLELE_CLASSES == 3


def test_wt_is_class_0():
    assert ALLELE_TO_CLASS["WT"] == 0


def test_structural_mutants_are_class_1():
    assert ALLELE_TO_CLASS["R175H"] == 1
    assert ALLELE_TO_CLASS["Y220C"] == 1


def test_dna_contact_mutants_are_class_2():
    for allele in ["R248W", "R248Q", "R273H", "R282W", "G245S"]:
        assert ALLELE_TO_CLASS[allele] == 2, f"{allele} should be class 2"


def test_all_training_alleles_covered():
    training_alleles = ["WT", "R175H", "R273H", "R248W", "R248Q", "R282W", "Y220C"]
    for a in training_alleles:
        assert a in ALLELE_TO_CLASS, f"{a} missing from ALLELE_TO_CLASS"


def test_unknown_allele_defaults_to_wt_class():
    assert allele_label_to_class_id("UNKNOWN_ALLELE") == 0
    assert allele_label_to_class_id("unknown") == 0


def test_class_names_length():
    assert len(CLASS_NAMES) == N_ALLELE_CLASSES


def test_get_class_name():
    assert get_class_name(0) == "WT"
    assert get_class_name(1) == "structural"
    assert get_class_name(2) == "dna_contact"

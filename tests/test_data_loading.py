"""Contract tests for data loading modules.

These test that the public interfaces exist and have correct signatures,
NOT that the actual data downloading works (that requires network).
"""
import pytest
import inspect

from tsgnn.data import download, preprocess, allele, grn_construction, temporal


def test_download_module_has_required_functions():
    assert hasattr(download, "download_geo_dataset")
    assert callable(download.download_geo_dataset)


def test_preprocess_module_has_required_functions():
    assert hasattr(preprocess, "preprocess_scrna")
    assert callable(preprocess.preprocess_scrna)


def test_allele_module_has_required_functions():
    assert hasattr(allele, "generate_esm2_embeddings") or hasattr(allele, "load_esm2_embeddings")


def test_grn_module_has_required_functions():
    assert hasattr(grn_construction, "build_base_grn")
    assert callable(grn_construction.build_base_grn)


def test_temporal_module_has_required_functions():
    assert hasattr(temporal, "create_temporal_graphs") or hasattr(temporal, "bin_by_pseudotime")

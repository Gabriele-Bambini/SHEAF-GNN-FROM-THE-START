"""Tests for typed configuration schema."""
import pytest
from tsgnn.config import (
    TSGNNConfig, DataConfig, ModelConfig, LossConfig, TrainingConfig,
)


def test_default_config_validates():
    config = TSGNNConfig()
    config.validate()  # should not raise


def test_invalid_K_raises():
    config = TSGNNConfig()
    config.data.K = 1
    with pytest.raises(ValueError, match="K must be"):
        config.validate()


def test_invalid_stalk_dim_raises():
    config = TSGNNConfig()
    config.model.stalk_dim = 1
    with pytest.raises(ValueError, match="stalk_dim must be"):
        config.validate()


def test_large_stalk_dim_warns():
    config = TSGNNConfig()
    config.model.stalk_dim = 16
    with pytest.warns(UserWarning, match="unusually large"):
        config.validate()


def test_high_lr_warns():
    config = TSGNNConfig()
    config.training.lr = 0.5
    with pytest.warns(UserWarning, match="very high"):
        config.validate()


def test_from_dict_roundtrip():
    config = TSGNNConfig(seed=123)
    d = config.to_dict()
    config2 = TSGNNConfig.from_dict(d)
    assert config2.seed == 123
    assert config2.model.stalk_dim == config.model.stalk_dim


def test_from_dict_ignores_unknown_keys():
    d = {"seed": 42, "model": {"stalk_dim": 4, "unknown_key": 999}}
    config = TSGNNConfig.from_dict(d)
    assert config.model.stalk_dim == 4


def test_negative_lambda_raises():
    config = TSGNNConfig()
    config.loss.lambda_1 = -0.1
    with pytest.raises(ValueError, match="lambda_1"):
        config.validate()

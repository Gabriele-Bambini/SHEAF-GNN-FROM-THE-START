"""Tests for the dependency-injected registries."""
import pytest
import torch.nn as nn
from tsgnn.model.registry import ModelRegistry, MetricRegistry


class DummyModel(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.fc = nn.Linear(10, hidden_dim)

    def forward(self, x):
        return self.fc(x)


def test_register_and_create():
    reg = ModelRegistry()
    reg.register("dummy", DummyModel)
    model = reg.create("dummy", hidden_dim=32)
    assert isinstance(model, DummyModel)
    assert model.fc.out_features == 32


def test_unknown_model_raises():
    reg = ModelRegistry()
    with pytest.raises(ValueError, match="Unknown model"):
        reg.create("nonexistent")


def test_list_models():
    reg = ModelRegistry()
    reg.register("b_model", DummyModel)
    reg.register("a_model", DummyModel)
    assert reg.list() == ["a_model", "b_model"]


def test_contains():
    reg = ModelRegistry()
    reg.register("dummy", DummyModel)
    assert "dummy" in reg
    assert "missing" not in reg


def test_metric_registry_run():
    mreg = MetricRegistry()
    mreg.register("add", lambda a, b: {"result": a + b})
    result = mreg.run("add", a=2, b=3)
    assert result == {"result": 5}


def test_metric_registry_run_all():
    mreg = MetricRegistry()
    mreg.register("double", lambda x: {"val": x * 2})
    mreg.register("square", lambda x: {"val": x ** 2})
    results = mreg.run_all(x=4)
    assert results["double"]["val"] == 8
    assert results["square"]["val"] == 16


def test_metric_run_all_handles_errors():
    mreg = MetricRegistry()
    mreg.register("failing", lambda: (_ for _ in ()).throw(ValueError("boom")))
    results = mreg.run_all()
    assert "error" in results["failing"]

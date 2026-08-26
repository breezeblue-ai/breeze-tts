import pytest
import torch

from breeze_infer.runtime import resolve_device, resolve_dtype


def test_explicit_device_wins_over_env(monkeypatch):
    monkeypatch.setenv("BREEZE_DEVICE", "cpu")
    assert resolve_device("mps") == "mps"


def test_env_device_is_used_when_no_explicit_device(monkeypatch):
    monkeypatch.setenv("BREEZE_DEVICE", "mps")
    assert resolve_device() == "mps"


def test_mps_is_preferred_over_cpu(monkeypatch):
    monkeypatch.delenv("BREEZE_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("breeze_infer.runtime.mps_is_available", lambda: True)
    assert resolve_device() == "mps"


def test_cuda_is_preferred_over_mps(monkeypatch):
    monkeypatch.delenv("BREEZE_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("breeze_infer.runtime.mps_is_available", lambda: True)
    assert resolve_device() == "cuda:0"


def test_cpu_falls_back_when_no_accelerator(monkeypatch):
    monkeypatch.delenv("BREEZE_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("breeze_infer.runtime.mps_is_available", lambda: False)
    assert resolve_device() == "cpu"


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cuda:0", torch.bfloat16), ("mps", torch.bfloat16), ("cpu", torch.float32)],
)
def test_default_dtype_per_device(monkeypatch, device, expected):
    monkeypatch.delenv("BREEZE_DTYPE", raising=False)
    assert resolve_dtype(device) == expected


def test_explicit_dtype_overrides_device_default(monkeypatch):
    monkeypatch.delenv("BREEZE_DTYPE", raising=False)
    assert resolve_dtype("mps", "float32") is torch.float32
    assert resolve_dtype("cpu", "bfloat16") is torch.bfloat16


def test_env_dtype_is_used(monkeypatch):
    monkeypatch.setenv("BREEZE_DTYPE", "float16")
    assert resolve_dtype("mps") is torch.float16


def test_unknown_dtype_raises(monkeypatch):
    monkeypatch.delenv("BREEZE_DTYPE", raising=False)
    with pytest.raises(ValueError, match="unsupported dtype"):
        resolve_dtype("mps", "float8")

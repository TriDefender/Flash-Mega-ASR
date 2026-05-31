from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


def load_optimized_linear_module():
    module_path = Path(__file__).resolve().parents[2] / "src" / "MegaASR" / "kernels" / "optimized_linear.py"
    spec = spec_from_file_location("optimized_linear_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


optimized_linear = load_optimized_linear_module()


def test_kernel_matmul_preserves_rank_3_linear_shape(monkeypatch):
    x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    weight_t = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    def fake_kernel(a, b, *, rmsnorm_weight=None, eps=1e-6):
        assert a.shape == (8, 3)
        assert b is weight_t
        assert rmsnorm_weight is None
        assert eps == 1e-6
        return a @ b

    monkeypatch.setattr(optimized_linear, "mega_kernel_fn", fake_kernel, raising=False)

    output = optimized_linear._kernel_matmul_preserve_leading_dims(x, weight_t)

    assert output.shape == (2, 4, 2)
    assert torch.allclose(output, torch.nn.functional.linear(x, weight_t.t()))


def test_kernel_matmul_preserves_rank_2_shape(monkeypatch):
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    weight_t = torch.ones(3, 2)

    def fake_kernel(a, b, *, rmsnorm_weight=None, eps=1e-6):
        assert a is x
        assert b is weight_t
        return a @ b

    monkeypatch.setattr(optimized_linear, "mega_kernel_fn", fake_kernel, raising=False)

    output = optimized_linear._kernel_matmul_preserve_leading_dims(x, weight_t)

    assert output.shape == (4, 2)
    assert torch.allclose(output, x @ weight_t)


def test_optimize_model_skips_lm_head():
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(4, 4)
    model.lm_head = torch.nn.Linear(4, 100)

    optimized_linear.OPTIMIZED_KERNEL_AVAILABLE = True
    optimized_linear.optimize_model(model)

    assert isinstance(model.proj, optimized_linear.OptimizedLinear)
    assert isinstance(model.lm_head, torch.nn.Linear)


def test_optimize_model_skips_very_wide_output_projection():
    model = torch.nn.Module()
    model.output_projection = torch.nn.Linear(4, 70000)

    optimized_linear.OPTIMIZED_KERNEL_AVAILABLE = True
    optimized_linear.optimize_model(model)

    assert isinstance(model.output_projection, torch.nn.Linear)

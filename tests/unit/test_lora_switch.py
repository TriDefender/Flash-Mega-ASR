from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import save_file as safe_save_file


torch_zeros = getattr(torch, "zeros")
torch_tensor = getattr(torch, "tensor")
torch_allclose = getattr(torch, "allclose")


def load_lora_switch_module():
    module_path = Path(__file__).resolve().parents[2] / "src" / "MegaASR" / "model" / "utils" / "lora_switch.py"
    spec = spec_from_file_location("megaasr_lora_switch_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lora_switch = load_lora_switch_module()
LoRADeltaSwitch = lora_switch.LoRADeltaSwitch


class FanInFanOutModule(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Module()
        self.linear.weight = torch.nn.Parameter(torch_zeros(in_features, out_features))


def write_adapter(tmp_path: Path, *, a_matrix: torch.Tensor, b_matrix: torch.Tensor, fan_in_fan_out: bool = False) -> Path:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": int(a_matrix.shape[0]),
                "lora_alpha": 8,
                "fan_in_fan_out": fan_in_fan_out,
            }
        ),
        encoding="utf-8",
    )
    safe_save_file(
        {
            "linear.lora_A.weight": a_matrix,
            "linear.lora_B.weight": b_matrix,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    return adapter_dir


def test_add_adapter_loads_deltas(tmp_path):
    model = torch.nn.Module()
    model.linear = torch.nn.Linear(3, 2, bias=False)
    switch = LoRADeltaSwitch(keep_delta_on_gpu=False)
    a_matrix = torch_tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]])
    b_matrix = torch_tensor([[2.0, -1.0], [1.5, 3.0]])
    adapter_dir = write_adapter(tmp_path, a_matrix=a_matrix, b_matrix=b_matrix)

    switch.add_adapter(model, adapter_dir, name="test-adapter")

    assert len(switch.items) == 1
    item = switch.items[0]
    expected_delta = torch.matmul(b_matrix, a_matrix) * 4.0
    assert item["name"] == "test-adapter"
    assert item["module_name"] == "linear"
    assert item["delta"].shape == model.linear.weight.shape
    assert item["delta"].device.type == "cpu"
    weight_dtype: Any = cast(torch.Tensor, model.linear.weight).dtype
    assert torch_allclose(item["delta"], expected_delta.to(dtype=weight_dtype))


def test_fan_in_fan_out_delta_matches_baseline(tmp_path):
    model = FanInFanOutModule(in_features=3, out_features=2)
    switch = LoRADeltaSwitch(keep_delta_on_gpu=False)
    a_matrix = torch_tensor([[1.0, -2.0, 0.5], [3.0, 4.0, -1.0]])
    b_matrix = torch_tensor([[2.0, 1.0], [-1.5, 0.25]])
    adapter_dir = write_adapter(tmp_path, a_matrix=a_matrix, b_matrix=b_matrix, fan_in_fan_out=True)

    switch.add_adapter(model, adapter_dir, name="fan-in-fan-out")

    assert len(switch.items) == 1
    scaling = 4.0
    expected_delta = (torch.matmul(b_matrix, a_matrix) * scaling).T
    assert switch.items[0]["delta"].shape == model.linear.weight.shape
    weight_dtype: Any = cast(torch.Tensor, model.linear.weight).dtype
    assert torch_allclose(switch.items[0]["delta"], expected_delta.to(dtype=weight_dtype))

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace


def load_hub_module():
    module_path = Path(__file__).resolve().parents[2] / "src" / "MegaASR" / "model" / "hub.py"
    spec = spec_from_file_location("megaasr_hub_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hub = load_hub_module()


def test_partial_resolution_preserves_hf_model_repo_id(monkeypatch, tmp_path):
    snapshot_dir = tmp_path / "snapshot"
    monkeypatch.setattr(hub, "_resolve_mega_asr_snapshot", lambda: str(snapshot_dir))

    sources = hub.resolve_sources(model_path="Qwen/Qwen3-ASR-1.7B")

    assert sources["model_path"] == "Qwen/Qwen3-ASR-1.7B"
    assert sources["lora_dir"] == str(snapshot_dir / "mega-asr-merged")
    assert sources["router_checkpoint"] == str(snapshot_dir / "audio_quality_router" / "best_acc_model.safetensors")


def test_all_explicit_resolution_preserves_hf_model_repo_id(tmp_path):
    router_checkpoint = tmp_path / "audio_quality_router" / "best_acc_model.safetensors"

    sources = hub.resolve_sources(
        model_path="Qwen/Qwen3-ASR-1.7B",
        lora_dir=tmp_path / "mega-asr-merged",
        router_checkpoint=router_checkpoint,
    )

    assert sources["model_path"] == "Qwen/Qwen3-ASR-1.7B"
    assert sources["lora_dir"] == str(tmp_path / "mega-asr-merged")
    assert sources["router_checkpoint"] == str(router_checkpoint)


def test_download_all_assets_downloads_full_local_layout(monkeypatch, tmp_path):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return kwargs["local_dir"]

    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    target_dir = tmp_path / "Mega-ASR"

    assert hub.download_all_assets(target_dir) == str(target_dir)
    assert calls == [
        {
            "repo_id": "Qwen/Qwen3-ASR-1.7B",
            "repo_type": "model",
            "local_dir": str(target_dir / "Qwen3-ASR-1.7B"),
            "local_dir_use_symlinks": False,
        },
        {
            "repo_id": "zhifeixie/Mega-ASR",
            "repo_type": "model",
            "allow_patterns": ["mega-asr-merged/*", "audio_quality_router/*"],
            "local_dir": str(target_dir),
            "local_dir_use_symlinks": False,
        },
    ]

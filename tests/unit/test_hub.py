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


# ---- _is_repo_id edge cases ------------------------------------------------ #


class TestIsRepoId:
    """Validate the repo-ID vs local-path heuristic."""

    def test_standard_repo_id(self):
        assert hub._is_repo_id("Qwen/Qwen3-ASR-1.7B") is True

    def test_mega_asr_repo_id(self):
        assert hub._is_repo_id("zhifeixie/Mega-ASR") is True

    def test_nested_local_path(self):
        """Multi-segment local paths are NOT repo IDs."""
        assert hub._is_repo_id("tmp/nonexistent/model") is False

    def test_deep_nested_local_path(self):
        assert hub._is_repo_id("a/b/c/d") is False

    def test_relative_dot_slash(self):
        assert hub._is_repo_id("./local/model") is False

    def test_relative_dot_dot_slash(self):
        assert hub._is_repo_id("../parent/model") is False

    def test_absolute_path(self):
        assert hub._is_repo_id("/abs/path") is False

    def test_windows_backslash(self):
        assert hub._is_repo_id("C:\\Users\\model") is False

    def test_bare_name_no_slash(self):
        assert hub._is_repo_id("just-a-name") is False

    def test_empty_string(self):
        assert hub._is_repo_id("") is False

    def test_trailing_slash(self):
        assert hub._is_repo_id("org/") is False

    def test_leading_slash(self):
        assert hub._is_repo_id("/name") is False


# ---- resolve_sources: local-path and ckpt_dir default flows ---------------- #


class TestResolveLocalPaths:
    """Ensure local paths are never mangled into repo IDs."""

    def test_ckpt_dir_overrides_everything(self, tmp_path):
        ckpt = tmp_path / "checkpoints" / "Mega-ASR"
        sources = hub.resolve_sources(ckpt_dir=ckpt)
        assert sources["model_path"] == str(ckpt / "Qwen3-ASR-1.7B")
        assert sources["lora_dir"] == str(ckpt / "mega-asr-merged")
        assert sources["router_checkpoint"] == str(
            ckpt / "audio_quality_router" / "best_acc_model.safetensors"
        )

    def test_ckpt_dir_with_routing_disabled(self, tmp_path):
        ckpt = tmp_path / "checkpoints" / "Mega-ASR"
        sources = hub.resolve_sources(ckpt_dir=ckpt, routing_enabled=False)
        assert sources["model_path"] == str(ckpt / "Qwen3-ASR-1.7B")
        assert sources["router_checkpoint"] is None

    def test_explicit_local_model_path_not_mangled(self, tmp_path):
        """A non-existent local path with '/' must NOT be treated as a repo ID."""
        model_dir = str(tmp_path / "nonexistent" / "model")
        sources = hub.resolve_sources(
            model_path=model_dir,
            lora_dir=str(tmp_path / "lora"),
            router_checkpoint=str(tmp_path / "router.safetensors"),
        )
        # Should be the expanded local path, not treated as a repo ID
        assert sources["model_path"] == model_dir

    def test_auto_detect_default_local_layout(self, monkeypatch, tmp_path):
        """If ckpt/Mega-ASR exists locally and no model_path given, use it automatically."""
        local_ckpt = tmp_path / "ckpt" / "Mega-ASR"
        local_ckpt.mkdir(parents=True)

        # chdir so that the relative path "ckpt/Mega-ASR" resolves under tmp_path
        monkeypatch.chdir(tmp_path)

        sources = hub.resolve_sources()

        # resolve_sources returns paths derived from the relative ckpt_dir
        assert "Qwen3-ASR-1.7B" in sources["model_path"]
        assert "mega-asr-merged" in sources["lora_dir"]
        assert "best_acc_model.safetensors" in sources["router_checkpoint"]

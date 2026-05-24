# Flash-Mega-ASR

**Performance-optimized inference framework for [Mega-ASR](https://huggingface.co/zhifeixie/Mega-ASR)**, built on [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).

What Flash-Mega-ASR adds on top of Mega-ASR:

- **Zero-overhead LoRA switching** — precomputed weight deltas applied via in-place `weight.data.add_()`, no PEFT dispatch
- **Batched grouped inference** — route once, group by decision, batch per group
- **Auto flash-attention backend** — FA2 → FA3 → SDPA → eager, device-aware
- **Device & dtype auto-detection** — bf16/fp16/fp32 + CUDA/MPS/CPU, zero config
- **CLI + WebUI** — `flash-mega-asr` command and Streamlit demo

> For Mega-ASR's architecture (audio quality routing, LoRA dispatch), see the [upstream project](https://huggingface.co/zhifeixie/Mega-ASR).

## Installation

Package manager: [uv](https://docs.astral.sh/uv/) (recommended) or pip.

```bash
# Core install (editable)
uv pip install -e .

# FlashAttention 2 (recommended for CUDA)
uv pip install -U flash-attn --no-build-isolation

# Extras
uv pip install -e ".[webui]"    # Streamlit WebUI
uv pip install -e ".[eval]"     # WER/CER evaluation
uv pip install -e ".[all]"      # Everything
```

Requires Python ≥ 3.10, PyTorch ≥ 2.10, CUDA (recommended).

## Download Checkpoints

```bash
python scripts/download.py
```

Downloads all weights to `ckpt/Mega-ASR/`:

```
ckpt/Mega-ASR/
├── Qwen3-ASR-1.7B/                          # Base ASR model
├── mega-asr-merged/                          # LoRA adapter (adapter_model.safetensors)
└── audio_quality_router/
    └── best_acc_model.safetensors            # Quality classifier checkpoint
```

## Usage

### Command Line

```bash
# Single file transcription
flash-mega-asr --file-name audio.wav

# Batch processing with custom batch size
flash-mega-asr --files audio1.wav audio2.wav audio3.wav --batch-size 16

# With timestamps and specific device
flash-mega-asr --file-name audio.wav --timestamps --device cuda:0

# Disable routing (always use LoRA)
flash-mega-asr --file-name audio.wav --no-routing

# Custom checkpoint directory
flash-mega-asr --file-name audio.wav --ckpt-dir ./ckpt/Mega-ASR

# Print resolved runtime info (backend, device, dtype)
flash-mega-asr --backend-report
```

Output is written to `output.json` by default (configurable via `--transcript-path`). Each result includes the transcription text, route decision, degraded probability, backend used, and timing metadata.

### Python API

```python
from MegaASR import MegaASR

model = MegaASR(
    model_path="ckpt/Mega-ASR/Qwen3-ASR-1.7B",
    lora_dir="ckpt/Mega-ASR/mega-asr-merged",
    router_checkpoint="ckpt/Mega-ASR/audio_quality_router/best_acc_model.safetensors",
    routing_enabled=True,
    quality_threshold=0.5,
)

# Single inference with route info
result = model.infer("audio.wav", return_route=True)
print(result["text"])
print(f"LoRA activated: {result['use_lora']} (degraded prob: {result['degraded_prob']:.3f})")

# Force LoRA on/off
text = model.infer_with_lora("noisy.wav", language="English")
text = model.infer_without_lora("clean.wav", language="English")

# Batched grouped inference
results = model.batch_infer(["clean.wav", "noisy.wav", "reverb.wav"])
print(model.stats)  # {"total": 3, "use_base": 1, "use_lora": 2}
```

### Standalone Script

```bash
python infer.py --audio audio.wav --ckpt_dir ckpt/Mega-ASR
python infer.py --audio audio.wav --no-routing
```

Works from a fresh checkout without `uv pip install -e .` — the script bootstraps the `src/` path automatically.

### WebUI

```bash
streamlit run webui.py
```

Provides mic recording, file upload, real-time spectrogram visualization, and system resource monitoring. Supports English, Chinese, and Japanese interface languages.

## Benchmark

Batch inference throughput comparison against the original Mega-ASR. All tests run on a single **NVIDIA GeForce RTX 4060 Ti** with batch size 4, `max_new_tokens=256`, 6 audio samples (53.8s total audio), 3 inference repeats each.

### Batch Inference (batch_size=4)

| Engine | Precision | Attention | Batch Latency | Batch RTF | vs Mega-ASR |
|---|---|---|---|---|---|
| Mega-ASR (original) | float32 | SDPA | 4.04s | 0.0899 | — |
| **Flash-Mega-ASR** | **bf16** | **FlashAttn 2** | **2.67s** | **0.0594** | **1.51× faster** |
| **Flash-Mega-ASR** | **fp16** | **FlashAttn 2** | **2.80s** | **0.0623** | **1.44× faster** |
| **Flash-Mega-ASR** | **fp16** | **SDPA** | **2.42s** | **0.054** | **1.67× faster** |

### Model Load Time

| Engine | Load Time | Speedup |
|---|---|---|
| Mega-ASR (original) | 37.77s | — |
| **Flash-Mega-ASR** (all configs) | **~8.7s** | **~4.3× faster** |

The load time improvement comes from zero-copy LoRA delta precomputation instead of PEFT's full adapter dispatch.

### Per-Sample Latency

| Sample | Duration | Mega-ASR | Flash bf16+FA2 | Flash fp16+SDPA |
|---|---|---|---|---|
| sample-1 | 17.02s | 1.78s | 1.94s | 1.62s |
| sample-2 | 17.24s | 1.14s | 1.23s | 1.16s |
| sample-3 | 6.27s | 0.48s | 0.57s | 0.55s |
| sample-4 | 4.39s | 0.48s | 0.53s | 0.50s |
| sample-5 | 6.27s | 1.15s | 1.22s | 1.10s |
| sample-6 | 2.60s | 0.31s | 0.38s | 0.34s |

Single-sample latency is comparable between engines — the major gains come from batched grouped inference, which avoids repeated LoRA load/unload cycles and amortizes routing cost across the entire batch.

## Project Structure

```
src/MegaASR/
├── __init__.py                 # Package exports: MegaASR, TranscriptionResult, BatchTranscriptionResult
├── cli.py                      # CLI entry point (flash-mega-asr command)
├── model/
│   ├── megaASR.py              # MegaASR orchestrator: routing + LoRA switching + ASR
│   ├── Qwen3_ASR.py            # Qwen3-ASR wrapper (wraps qwen_asr package)
│   ├── router.py               # AudioQualityRouter: mel spectrogram → degraded classification
│   └── utils/
│       ├── audio_quality.py    # LogMelSpectrogram, AudioQualityClassifier (ConvFrontend + Transformer)
│       └── lora_switch.py      # LoRADeltaSwitch: precompute deltas, in-place weight switching
├── runtime/
│   ├── backend.py              # Attention backend resolver (FA2/FA3/SDPA/eager)
│   ├── device.py               # Device and dtype auto-detection
│   └── results.py              # TranscriptionResult / BatchTranscriptionResult dataclasses
├── A2S-SFT/                    # Supervised fine-tuning code
├── DG-WGPO/                    # RL training (coming soon)
├── eval/                       # WER/CER evaluation scripts
└── data/                       # Dataset download utilities
```

## Key Components

| Component | File | What it does |
|---|---|---|
| `LoRADeltaSwitch` | `model/utils/lora_switch.py` | Precomputes `delta = B @ A * α/rank`, applies via in-place `add_()` — sub-ms switching |
| `AudioQualityRouter` | `model/router.py` | Compact Transformer (1L, 192-dim) classifies audio as clean/degraded from log-mel |
| Backend Resolver | `runtime/backend.py` | Auto-selects FA2 → FA3 → SDPA → eager based on device |
| Device/Dtype | `runtime/device.py` | Auto-detects CUDA/MPS/CPU and bf16/fp16/fp32 |

## Citation

If you use this work, please cite both Mega-ASR and Qwen3-ASR:

```bibtex
@article{MegaASR,
  title={Mega-ASR: Robust Speech Recognition via Audio-Quality-Aware LoRA Routing},
  author={Zhifei Xie and Flash-Mega-ASR Contributors},
  year={2025}
}

@article{Qwen3-ASR,
  title={Qwen3-ASR Technical Report},
  author={Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, Jin Xu, Jingren Zhou, Junyang Lin},
  journal={arXiv preprint arXiv:2601.21337},
  year={2026}
}
```

## Acknowledgments

- [Mega-ASR](https://huggingface.co/zhifeixie/Mega-ASR) — the original LoRA routing system for robust ASR
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) — the base speech recognition model
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — fast and memory-efficient attention

## License

Apache-2.0

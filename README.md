# Flash-Mega-ASR

**Performance-optimized inference framework for [Mega-ASR](https://huggingface.co/zhifeixie/Mega-ASR).**

Mega-ASR is an audio-quality-aware speech recognition system that uses a lightweight classifier to detect degraded audio (noise, reverberation, far-field, clipping, etc.) and dynamically activates LoRA-enhanced weights only when needed. It is built on top of [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).

**Flash-Mega-ASR** is our modified and extended version that adds:

- **Zero-overhead LoRA switching** — Precomputed weight deltas are applied directly to model parameters via in-place addition (`weight.data.add_(delta)`), avoiding the PEFT load/unload overhead entirely. Switching takes sub-millisecond instead of seconds.
- **Batched grouped inference** — All audios are routed first, grouped by decision (base vs. LoRA), then processed in a single batched transcribe call per group. Minimizes the number of LoRA switches per batch.
- **Auto flash-attention backend** — Resolves the fastest available attention implementation at runtime (FlashAttention 2/3 → PyTorch SDPA → eager) based on hardware capability.
- **Device and dtype auto-detection** — Picks the optimal precision (bfloat16 on Ampere+, float16 on older CUDA, float32 on CPU) and device (CUDA → MPS → CPU) automatically.
- **CLI with JSON output** — `flash-mega-asr` command-line tool with progress bars, structured JSON results, and runtime metadata.
- **Streamlit WebUI** — Interactive demo with mic recording, file upload, real-time spectrograms, and system monitoring.

## How It Works

```
                          ┌─────────────────────────────┐
 Audio ──► AudioQualityRouter (192-dim Transformer)     │
            classifies as "clean" or "degraded"          │
                          │                              │
              ┌───────────┴───────────┐                  │
              ▼                       ▼                  │
        Qwen3-ASR              Qwen3-ASR + LoRA         │
        (base weights)         (delta added in-place)    │
              │                       │                  │
              └───────────┬───────────┘                  │
                          ▼                              │
                     Transcription                       │
                          └─────────────────────────────┘
```

The audio quality router is a compact Transformer encoder (1 layer, 192 dimensions) with a ConvFrontend that takes log-mel spectrograms and outputs a 2-class probability. When the "degraded" probability exceeds a configurable threshold, the system applies the precomputed LoRA delta to the base model weights. On clean audio, it skips the delta entirely.

The `LoRADeltaSwitch` precomputes `delta = B @ A * (alpha / rank)` at load time for every LoRA target module, then keeps the delta tensors on GPU. Switching is a single `weight.data.add_(delta, alpha=±1)` call per module — no PEFT dispatch, no adapter loading, no recomputation.

## Installation

```bash
# Core install (editable)
pip install -e .

# FlashAttention 2 (recommended for CUDA, significantly faster)
pip install -U flash-attn --no-build-isolation

# WebUI dependencies
pip install -e ".[webui]"

# Evaluation dependencies (WER/CER)
pip install -e ".[eval]"

# Everything
pip install -e ".[all]"
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

Works from a fresh checkout without `pip install -e .` — the script bootstraps the `src/` path automatically.

### WebUI

```bash
streamlit run webui.py
```

Provides mic recording, file upload, real-time spectrogram visualization, and system resource monitoring. Supports English, Chinese, and Japanese interface languages.

## Training

### A2S-SFT Fine-tuning

Progressive LoRA fine-tuning on the Qwen3-ASR base model with configurable scopes:

```
Stage 1: encoder_aligner  — adapt speech encoder + audio-text aligner
Stage 2: llm              — adapt the language model head
Stage 3: all              — joint optimization
```

```bash
bash scripts/finetune.sh
```

See [src/MegaASR/A2S-SFT/readme.md](src/MegaASR/A2S-SFT/readme.md) for details on LoRA scopes, per-module learning rates, and stage transitions.

### DG-WGPO Reinforcement Learning

Reinforcement learning pipeline for LoRA training (coming soon). See [src/MegaASR/DG-WGPO/README.md](src/MegaASR/DG-WGPO/README.md).

## Evaluation

```bash
python src/MegaASR/eval/evaluate_wer.py \
  --ckpt_dir ckpt/Mega-ASR \
  --input_jsonl test.jsonl \
  --output_jsonl results.jsonl
```

Input is JSONL with `audio` and `answer` fields. The script appends `prediction`, `wer`, `metric` (WER for English, CER for Chinese), `num_edits`, and `ref_len` to each record.

See [src/MegaASR/eval/readme.md](src/MegaASR/eval/readme.md) for format details.

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

### LoRADeltaSwitch

At initialization, loads the LoRA adapter safetensors and computes `delta = (B @ A) * alpha / rank` for every target module. Stores deltas on GPU. Switching between base and LoRA is a single pass of `weight.data.add_(delta, alpha=±1)` across all modules — no PEFT dispatch overhead, no adapter re-injection.

Supports block-structured LoRA (`mega_lora_blocks.json`) where different blocks of the same weight matrix have different ranks and alpha values.

### AudioQualityRouter

A compact audio classifier: `ConvFrontend (2-layer 1D Conv)` → `PositionalEncoding` → `TransformerEncoder (1 layer)` → `AttentionPooling` → `Linear classifier (2-class)`.

Takes 16kHz mono audio, computes log-mel spectrograms (80 bins), runs a single forward pass. Returns `(is_degraded: bool, degraded_prob: float)`. Supports batch inference with automatic length padding and masking.

### Backend Resolver

Auto-detects the best attention implementation:
- **FlashAttention 2** — fastest on CUDA, requires `flash-attn` package
- **FlashAttention 3** — Hopper (H100+) GPUs only
- **PyTorch SDPA** — built-in scaled dot product attention (torch ≥ 2.0)
- **Eager** — fallback, no kernel optimizations

Resolution is device-aware: CUDA tries FA2 first, CPU/MPS goes straight to SDPA.

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

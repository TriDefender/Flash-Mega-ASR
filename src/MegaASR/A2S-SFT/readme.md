## A2S-SFT Training

`src/MegaASR/A2S-SFT` contains the core training code for Mega-ASR supervised fine-tuning. It is designed for Qwen3-ASR-style speech-to-text models and supports LoRA training on different parts of the model.

```text
src/MegaASR/A2S-SFT/
├── arguments.py      # Defines training arguments and hyperparameters.
├── checkpointing.py  # Saves base-model metadata and processor/tokenizer files for LoRA reuse.
├── dataloader.py     # Loads JSONL data, reads audio, builds inputs, and masks non-target labels.
├── finetune.py       # Main entry point for launching A2S-SFT training.
├── modeling.py       # Loads the model and defines LoRA injection scopes.
├── trainer.py        # Defines MegaASRTrainer with adapter-only saving and module-wise learning rates.
```

### Model and LoRA Scope

Choose the base ASR model with `--model_path`. The LoRA training range is controlled by `--lora_scope`:

```text
encoder            # speech encoder only
aligner            # audio-text aligner / projector only
encoder_aligner    # speech encoder + aligner
encoder_b4_aligner # last four encoder layers + aligner
llm                # language model only
all                # encoder + aligner + language model
```

In our training pipeline, we use a progressive strategy:

```text
Stage 1: encoder_aligner
  First adapt the speech encoder and audio-text aligner for robust acoustic perception and alignment.

Stage 2: llm
  Then adapt the LLM to improve transcription generation under degraded acoustic conditions.

Stage 3: all
  Finally tune encoder, aligner, and LLM together for joint optimization.
```

The learning rates of the three parts can be set separately:

```text
--lr_encoder  # learning rate for speech encoder LoRA
--lr_aligner  # learning rate for audio-text aligner LoRA
--lr_llm      # learning rate for LLM LoRA
```

If a later stage starts from a previous LoRA checkpoint, use:

```bash
--merge_lora_into_base_from ${PREVIOUS_LORA_DIR}
```
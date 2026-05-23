## ASR Evaluation

We provide a simple evaluation script for running Mega-ASR inference and computing WER/CER.  
The input file should be a JSONL file. Each line only needs two required fields:

```json
{"audio": "examples/audio/noise.wav", "answer": "I usually take the quieter road home because the main street gets crowded after work."}
```


The script will keep all original fields and append the following fields to the output JSONL:

```text
prediction  # model transcription
metric      # "wer" for English samples, "cer" for Chinese samples
wer         # WER/CER score value; CER is also stored in this field for compatibility
num_edits   # edit distance between prediction and ground truth
ref_len     # number of reference words or characters
```

The script reuses the Mega-ASR inference wrapper, so it loads the base Qwen3-ASR model,
the Mega-ASR LoRA, and the router from the checkpoint directory:

```text
ckpt/Mega-ASR/
├── Qwen3-ASR-1.7B
├── mega-asr-merged
└── audio_quality_router/best_acc_model.safetensors
```

### Run Evaluation

```bash
python src/MegaASR/eval/evaluate_wer.py \
  --ckpt_dir ckpt/Mega-ASR \
  --input_jsonl examples/test.jsonl \
  --output_jsonl outputs/pred_with_wer.jsonl
```

Disable routing if you want to always use the Mega-ASR LoRA:

```bash
python src/MegaASR/eval/evaluate_wer.py \
  --ckpt_dir ckpt/Mega-ASR \
  --input_jsonl examples/test.jsonl \
  --output_jsonl outputs/pred_with_wer.jsonl \
  --no-routing
```

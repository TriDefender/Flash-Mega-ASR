# coding=utf-8
import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path
from tqdm import tqdm

BATCH_SIZE = 8
MAX_NEW_TOKENS = 256
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from cn_tn import TextNorm
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

ENGLISH_NORMALIZER = EnglishTextNormalizer()
CHINESE_NORMALIZER = TextNorm(
    to_banjiao=False,
    to_upper=False,
    to_lower=False,
    remove_fillers=False,
    remove_erhua=False,
    check_chars=False,
    remove_space=False,
    cc_mode="",
)
BASIC_NORMALIZER = BasicTextNormalizer()

def detect_language(ref, pred):
    return "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in ref + pred) else "en"

def unwrap_prediction(pred):
    if isinstance(pred, list):
        return " ".join(str(x) for x in pred)
    return str(pred)

class EvaluationTokenizer:
    SPACE = chr(32)
    SPACE_ESCAPE = chr(9601)

    def __init__(
        self,
        tokenizer_type: str = "zh",
        lowercase: bool = True,
        punctuation_removal: bool = True,
        character_tokenization: bool = False,
    ):
        from sacrebleu.tokenizers import TOKENIZERS

        self.tokenizer = TOKENIZERS[tokenizer_type]
        self.lowercase = lowercase
        self.punctuation_removal = punctuation_removal
        self.character_tokenization = character_tokenization

    @classmethod
    def remove_punctuation(cls, sent: str):
        return cls.SPACE.join(
            token
            for token in sent.split(cls.SPACE)
            if not all(unicodedata.category(char)[0] == "P" for char in token)
        )

    def tokenize(self, sent: str):
        tokenized = self.tokenizer()(sent)
        if self.punctuation_removal:
            tokenized = self.remove_punctuation(tokenized)
        if self.character_tokenization:
            tokenized = self.SPACE.join(list(tokenized.replace(self.SPACE, self.SPACE_ESCAPE)))
        if self.lowercase:
            tokenized = tokenized.lower()
        return tokenized

def compute_one_error(ref, pred, language):
    import editdistance as ed
    import zhconv

    if language == "yue":
        ref = zhconv.convert(ref, "zh-cn")
        pred = zhconv.convert(pred, "zh-cn")
    if language == "en":
        ref = ENGLISH_NORMALIZER(ref)
        pred = ENGLISH_NORMALIZER(pred)
    elif language == "zh":
        ref = CHINESE_NORMALIZER(ref)
        pred = CHINESE_NORMALIZER(pred)
    else:
        ref = BASIC_NORMALIZER(ref)
        pred = BASIC_NORMALIZER(pred)

    tokenizer = EvaluationTokenizer()
    ref_items = tokenizer.tokenize(ref).split()
    pred_items = tokenizer.tokenize(pred).split()
    edits = ed.eval(ref_items, pred_items)
    ref_len = len(ref_items)
    return (edits / ref_len if ref_len else 0.0), {"err": int(edits), "nref": int(ref_len)}

def resolve_audio(path, jsonl_path):
    path = Path(path)
    if path.is_absolute():
        return str(path)
    jsonl_dir_path = Path(jsonl_path).resolve().parent / path
    return str(jsonl_dir_path if jsonl_dir_path.exists() else Path.cwd() / path)

def get_audio_field(item):
    if "audio" in item:
        return item["audio"]
    if "audio_path" in item:
        return item["audio_path"]
    raise KeyError("Input JSONL item must contain `audio` or `audio_path`.")

def main():
    parser = argparse.ArgumentParser("Run Mega-ASR inference and compute WER/CER.")
    parser.add_argument("--ckpt_dir", required=True, help="Mega-ASR ckpt root")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--routing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--keep_delta_on_gpu", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from MegaASR.model.megaASR import MegaASR

    ckpt_dir = Path(args.ckpt_dir).expanduser()

    model = MegaASR(
        model_path=ckpt_dir / "Qwen3-ASR-1.7B",
        lora_dir=ckpt_dir / "mega-asr-merged",
        router_checkpoint=ckpt_dir / "audio_quality_router/best_acc_model.safetensors",
        routing_enabled=args.routing,
        quality_threshold=args.threshold,
        device_map=args.device_map,
        max_inference_batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        keep_delta_on_gpu=args.keep_delta_on_gpu,
    )
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    outputs, total_edits, total_ref_len = [], 0, 0

    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="evaluating"):
        batch = data[i:i + BATCH_SIZE]
        audio_paths = [resolve_audio(get_audio_field(x), args.input_jsonl) for x in batch]
        results = model.batch_infer(audio_paths)
        for item, pred in zip(batch, results):
            pred = unwrap_prediction(pred).strip()
            language = item.get("language") or detect_language(item["answer"], pred)
            score, detail = compute_one_error(item["answer"], pred, language)
            edits = detail["err"]
            ref_len = detail["nref"]
            metric = "cer" if language in {"zh", "yue"} else "wer"
            item["prediction"] = pred
            item["metric"] = metric
            item["wer"] = round(float(score), 6)
            item["num_edits"] = int(edits)
            item["ref_len"] = int(ref_len)
            total_edits += edits
            total_ref_len += ref_len
            outputs.append(item)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"samples: {len(outputs)}")
    print(f"overall_error: {total_edits / total_ref_len if total_ref_len else 0.0:.6f}")
    print(f"saved: {out_path}")

if __name__ == "__main__":
    main()

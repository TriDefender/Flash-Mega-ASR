import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "zhifeixie/Mega-ASR"
DEFAULT_BASE_REPO_ID = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_LOCAL_DIR = ROOT_DIR / "ckpt/Mega-ASR"
sys.path.insert(0, str(ROOT_DIR / "src"))

from MegaASR.model.hub import download_all_assets  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Download Mega-ASR weights")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID, help="Mega-ASR Hugging Face repo id")
    parser.add_argument("--base_repo_id", default=DEFAULT_BASE_REPO_ID, help="Base ASR Hugging Face repo id")
    parser.add_argument("--local_dir", default=DEFAULT_LOCAL_DIR, help="local ckpt dir")
    return parser.parse_args()


def main():
    args = parse_args()
    local_dir = download_all_assets(
        args.local_dir,
        base_model_repo=args.base_repo_id,
        mega_asr_repo=args.repo_id,
    )
    print(f"Downloaded to {local_dir}")


if __name__ == "__main__":
    main()

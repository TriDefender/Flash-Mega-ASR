import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "zhifeixie/Mega-ASR"
DEFAULT_LOCAL_DIR = ROOT_DIR / "ckpt/Mega-ASR"


def parse_args():
    parser = argparse.ArgumentParser(description="Download Mega-ASR weights")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID, help="Hugging Face repo id")
    parser.add_argument("--local_dir", default=DEFAULT_LOCAL_DIR, help="local ckpt dir")
    return parser.parse_args()


def main():
    args = parse_args()
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=str(args.local_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to {args.local_dir}")


if __name__ == "__main__":
    main()

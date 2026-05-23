#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON_BIN" infer.py "$@"

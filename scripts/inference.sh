#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON_BIN" infer.py "$@"

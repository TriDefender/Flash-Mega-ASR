"""Tests for CLI argument parsing."""
from MegaASR.cli import parse_args


class TestParseArgs:
    def test_file_name(self):
        args = parse_args(["--file-name", "test.wav"])
        assert args.file_name == "test.wav"

    def test_batch_files(self):
        args = parse_args(["--files", "a.wav", "b.wav", "c.wav"])
        assert args.files == ["a.wav", "b.wav", "c.wav"]

    def test_attn_backend(self):
        args = parse_args(["--file-name", "t.wav", "--attn", "sdpa"])
        assert args.attn == "sdpa"

    def test_defaults(self):
        args = parse_args(["--file-name", "t.wav"])
        assert args.batch_size == 24
        assert args.attn == "auto"
        assert args.routing is True
        assert args.threshold == 0.5
        assert args.max_new_tokens == 128

    def test_no_routing(self):
        args = parse_args(["--file-name", "t.wav", "--no-routing"])
        assert args.routing is False

#!/usr/bin/env python3
"""Stable local Whisper command used by TARS messaging gateways.

The gateway runtime can be newer than the Python versions supported by
``ctranslate2``.  Keeping STT in a small, dedicated Python 3.11 environment
lets gateway upgrades remain independent from voice transcription.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--model", default="base")
    parser.add_argument("--output_format", "--output-format", default="txt")
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    parser.add_argument("--language", default="pt")
    return parser


def transcribe(input_path: Path, model_name: str, language: str) -> str:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        str(input_path),
        language=language,
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def main() -> int:
    args = _parser().parse_args()
    if args.output_format != "txt":
        raise SystemExit("TARS local STT supports only txt output")
    if not args.input_path.is_file():
        raise SystemExit(f"Audio file not found: {args.input_path}")

    transcript = transcribe(args.input_path, args.model, args.language)
    if not transcript:
        raise SystemExit("Local Whisper returned an empty transcript")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.input_path.stem}.txt"
    output_path.write_text(transcript + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

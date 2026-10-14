from pathlib import Path

from scripts import tars_local_stt


def test_cli_writes_portuguese_transcript_for_gateway_contract(tmp_path, monkeypatch):
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"valid-enough-for-the-mocked-transcriber")
    output_dir = tmp_path / "output"
    seen = {}

    def fake_transcribe(input_path: Path, model_name: str, language: str) -> str:
        seen.update(input_path=input_path, model=model_name, language=language)
        return "Áudio entendido."

    monkeypatch.setattr(tars_local_stt, "transcribe", fake_transcribe)
    monkeypatch.setattr(
        "sys.argv",
        [
            "tars_local_stt.py",
            str(audio),
            "--model",
            "base",
            "--output_format",
            "txt",
            "--output_dir",
            str(output_dir),
        ],
    )

    assert tars_local_stt.main() == 0
    assert seen == {"input_path": audio, "model": "base", "language": "pt"}
    assert (output_dir / "voice.txt").read_text(encoding="utf-8") == "Áudio entendido.\n"

"""Public local-audio service and JSON CLI."""

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence, Union

from .backend import SpeechBackend
from .diagnostics import platform_diagnostics
from .factory import BackendConfig, create_backend
from .metrics import summarize_transcript
from .models import SpeechError, TranscriptionResult, UnsupportedAudioError


SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff", ".aif"}


def transcribe_audio(path: Union[str, Path], backend: SpeechBackend) -> TranscriptionResult:
    """Validate a local audio path, run the supplied engine, and derive metrics."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise UnsupportedAudioError(
            f"Unsupported audio format '{path.suffix or '(none)'}'; "
            f"expected one of {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    try:
        if not path.is_file():
            raise SpeechError(f"Audio does not exist or is not a regular file: {path}")
        with path.open("rb") as audio:
            if not audio.read(1):
                raise SpeechError(f"Audio file is empty: {path.name}")
        return summarize_transcript(backend.transcribe(path))
    except SpeechError:
        raise
    except Exception as exc:
        raise SpeechError(f"Cannot transcribe '{path.name}': audio or backend failure") from exc


def main(argv: Optional[Sequence[str]] = None) -> int:
    cli = argparse.ArgumentParser(description="Transcribe local audio to JSON with an explicitly selected backend.")
    cli.add_argument("path", nargs="?", type=Path, help="Local audio file, e.g. sample_data/speech_demo.wav")
    cli.add_argument("--backend", default="cpu_whisper", help="cpu_whisper (default) or qnn_whisper (planned)")
    cli.add_argument("--model-path", type=Path, default=Path("models/whisper-tiny.en"),
                     help="Pre-downloaded CTranslate2 model directory")
    cli.add_argument("--language", default="en", help="Audio language code (default: en)")
    cli.add_argument("--model-name", help="Model label for metadata; defaults to the model directory name")
    inspection = cli.add_mutually_exclusive_group()
    inspection.add_argument("--backend-info", action="store_true", help="Print configured metadata without loading a model")
    inspection.add_argument("--diagnostics", action="store_true", help="Print platform/dependency diagnostics without inference")
    args = cli.parse_args(argv)
    if args.diagnostics:
        print(json.dumps(platform_diagnostics(), indent=2))
        return 0
    if args.path is None and not args.backend_info:
        cli.error("an audio path is required unless --backend-info or --diagnostics is used")
    try:
        backend = create_backend(BackendConfig(
            backend=args.backend, model_path=args.model_path,
            language=args.language, model_name=args.model_name,
        ))
        if args.backend_info:
            print(json.dumps(backend.metadata.to_dict(), indent=2))
            return 0
        result = transcribe_audio(args.path, backend)
    except SpeechError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

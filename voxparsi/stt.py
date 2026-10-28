"""Persian speech-to-text model catalogue.

The app keeps a small set of locally runnable Whisper variants and selects the
best option for the current device. The model list is intentionally compact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import whisper

WHISPER_CACHE = Path.home() / ".cache" / "whisper"
DEFAULT_MODEL = "small"


@dataclass(frozen=True)
class ModelSpec:
    """One selectable speech-to-text model."""

    key: str
    label: str
    backend: str
    ref: str
    note: str
    download_mb: int

    @property
    def is_cached(self) -> bool:
        """Whether the weights are already on disk, so selecting is free."""
        if self.backend == "whisper":
            return (WHISPER_CACHE / f"{self.ref}.pt").is_file()
        cache = Path.home() / ".cache" / "huggingface" / "hub"
        return (cache / f"models--{self.ref.replace('/', '--')}").is_dir()


CATALOGUE: tuple[ModelSpec, ...] = (
    ModelSpec(
        "small",
        "Whisper small (multilingual)",
        "whisper",
        "small",
        "Already downloaded.",
        461,
    ),
    ModelSpec(
        "turbo",
        "Whisper large-v3-turbo",
        "whisper",
        "turbo",
        "809M. Best accuracy that fits a 4 GB GPU; falls back to CPU if it does not.",
        1620,
    ),
    ModelSpec(
        "medium",
        "Whisper medium",
        "whisper",
        "medium",
        "769M. Slower than turbo for no accuracy gain; kept for comparison.",
        1530,
    ),
)

CATALOGUE_BY_KEY = {spec.key: spec for spec in CATALOGUE}


def resolve(key: str) -> ModelSpec:
    if key not in CATALOGUE_BY_KEY:
        raise KeyError(f"Unknown model {key!r}. Known: {sorted(CATALOGUE_BY_KEY)}")
    return CATALOGUE_BY_KEY[key]


def available_models() -> list[ModelSpec]:
    """The catalogue, already-downloaded models first."""
    return sorted(CATALOGUE, key=lambda spec: (not spec.is_cached, spec.download_mb))


def cached_models() -> list[str]:
    return [spec.key for spec in CATALOGUE if spec.is_cached]


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class TranscriptionOptions:
    """Decoding knobs, shared by both backends.

    `beam_size` and `temperature` interact: Whisper's own fallback loop drops
    beam search as soon as it retries at a non-zero temperature, so a beam size
    only takes effect while `temperature` is 0.
    """

    language: str = "fa"
    beam_size: int | None = 5
    temperature: float = 0.0
    initial_prompt: str | None = None


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str
    model: str
    device: str
    segments: list[dict[str, Any]] = field(default_factory=list)


class Transcriber:
    """A loaded speech-to-text model."""

    spec: ModelSpec
    device: str

    def transcribe(self, audio_path: str | Path, options: TranscriptionOptions) -> Transcription:
        raise NotImplementedError


class _WhisperTranscriber(Transcriber):
    def __init__(self, spec: ModelSpec, device: str) -> None:
        self.spec = spec
        self.device = device
        self.model = whisper.load_model(spec.ref, device=device)

    def transcribe(self, audio_path: str | Path, options: TranscriptionOptions) -> Transcription:
        kwargs: dict[str, Any] = {
            "language": options.language,
            "temperature": options.temperature,
            "fp16": self.device == "cuda",
        }
        if options.beam_size:
            kwargs["beam_size"] = options.beam_size
        if options.initial_prompt:
            kwargs["initial_prompt"] = options.initial_prompt

        result = self.model.transcribe(str(audio_path), **kwargs)
        return Transcription(
            text=str(result.get("text", "")).strip(),
            language=str(result.get("language", options.language)),
            model=self.spec.key,
            device=self.device,
            segments=list(result.get("segments", [])),
        )


class _TransformersTranscriber(Transcriber):
    """A Hugging Face Whisper checkpoint, for the Persian fine-tunes."""

    def __init__(self, spec: ModelSpec, device: str) -> None:
        from transformers import pipeline

        self.spec = spec
        self.device = device
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=spec.ref,
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            chunk_length_s=30,
        )

    def transcribe(self, audio_path: str | Path, options: TranscriptionOptions) -> Transcription:
        generate_kwargs: dict[str, Any] = {
            "language": options.language,
            "task": "transcribe",
        }
        if options.beam_size and options.temperature == 0:
            generate_kwargs["num_beams"] = options.beam_size

        result = self.pipe(str(audio_path), generate_kwargs=generate_kwargs)
        return Transcription(
            text=str(result.get("text", "")).strip(),
            language=options.language,
            model=self.spec.key,
            device=self.device,
        )


def load_model(key: str = DEFAULT_MODEL, device: str = "auto") -> Transcriber:
    """Load a catalogue model, retrying on CPU if the GPU cannot hold it.

    The fallback is the point of this function: `turbo` is recommended precisely
    because it usually fits 4 GB, and "usually" has to degrade to a slow
    transcription rather than a crash mid-experiment.
    """
    spec = resolve(key)
    chosen = pick_device(device)
    backend = _WhisperTranscriber if spec.backend == "whisper" else _TransformersTranscriber

    try:
        return backend(spec, chosen)
    except torch.cuda.OutOfMemoryError:
        if chosen != "cuda":
            raise
        torch.cuda.empty_cache()
        return backend(spec, "cpu")


def transcribe(
    model: Transcriber,
    audio_path: str | Path,
    options: TranscriptionOptions | None = None,
) -> Transcription:
    """Transcribe one audio file."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"No such audio file: {path}")
    return model.transcribe(path, options or TranscriptionOptions())

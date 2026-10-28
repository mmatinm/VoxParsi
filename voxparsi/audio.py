"""Audio helpers for Persian speech-to-text evaluation.

The module reads waveform files, assembles long speaker references, and keeps the
STT flow independent from optional model-specific dependencies.
"""

from __future__ import annotations

import io
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_VOICE_PROMPT_SEC = 5.0

REFERENCE_TARGET_SEC = 45.0
REFERENCE_MIN_SEC = 30.0

_INT16_FULL_SCALE = 32768.0


def _read_wav_stdlib(path: Path) -> tuple[np.ndarray, int] | None:
    """Decode a PCM WAV with the standard library, or None if unsupported."""
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getsampwidth() != 2 or handle.getcomptype() != "NONE":
                return None
            channels = handle.getnchannels()
            rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError, OSError):
        return None

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / _INT16_FULL_SCALE
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode any supported audio file to mono float32 samples and its rate."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such audio file: {path}")

    decoded = _read_wav_stdlib(path)

    if decoded is None:
        try:
            import soundfile as sf
        except ImportError as error:  # pragma: no cover - depends on install
            raise RuntimeError(
                f"{path.name} is not a 16-bit PCM WAV, so it needs soundfile: "
                "pip install soundfile"
            ) from error
        samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
        decoded = samples.mean(axis=1), int(rate)

    samples, rate = decoded
    if samples.size == 0:
        raise ValueError(f"{path.name} decoded to an empty signal.")
    return np.ascontiguousarray(samples, dtype=np.float32), rate


def write_temp_audio(data: bytes, suffix: str) -> Path:
    """Spill uploaded bytes to a file, because the decoders take paths."""
    with tempfile.NamedTemporaryFile(suffix=suffix or ".wav", delete=False) as handle:
        handle.write(data)
        return Path(handle.name)


def to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float samples as a 16-bit PCM WAV, for playback and download."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / peak
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def resample_mono(samples: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono audio, preferring a real anti-aliased implementation.

    The SciPy path is preferred when available; the linear fallback keeps the
    evaluation flow working without optional extra dependencies.
    """
    if sample_rate == target_rate:
        return samples

    try:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(sample_rate), int(target_rate))
        return np.asarray(
            resample_poly(samples, target_rate // divisor, sample_rate // divisor),
            dtype=np.float32,
        )
    except ImportError:
        duration = samples.size / sample_rate
        target_positions = np.linspace(
            0.0, duration, int(duration * target_rate), endpoint=False, dtype=np.float64
        )
        source_positions = np.arange(samples.size, dtype=np.float64) / sample_rate
        return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _frame_rms(samples: np.ndarray, window: int) -> np.ndarray:
    """RMS per non-overlapping window, as a 1-D array."""
    usable = (samples.size // window) * window
    if usable == 0:
        return np.zeros(0, dtype=np.float64)
    frames = samples[:usable].astype(np.float64).reshape(-1, window)
    return np.sqrt((frames**2).mean(axis=1))


def trim_silence(samples: np.ndarray, sample_rate: int, keep_ms: float = 120.0) -> np.ndarray:
    """Cut leading and trailing silence, keeping a short margin.

    Used for two different reasons. On a voice prompt, silence wastes part of a
    budget that should be carrying the speaker's voice. On a generated chunk,
    every generation opens with silence of its own, so concatenating untrimmed
    chunks puts an audible hole mid-phrase that no pause setting controls,
    because nothing inserted it.

    The threshold is relative to the signal's own RMS and the margin is
    generous: a voiced stop is brief and quiet, so a tighter detector locks onto
    the following vowel and clips the consonant.
    """
    if samples.size == 0:
        return samples
    rms = float(np.sqrt((samples.astype(np.float64) ** 2).mean()))
    if rms <= 0:
        return samples

    window = max(1, int(0.02 * sample_rate))
    energies = _frame_rms(samples, window)
    loud = np.flatnonzero(energies > 0.04 * rms)
    if loud.size == 0:
        return samples

    margin = int(keep_ms / 1000.0 * sample_rate)
    start = max(0, loud[0] * window - margin)
    end = min(samples.size, (loud[-1] + 1) * window + margin)
    return samples[start:end]


def best_windows(
    samples: np.ndarray,
    sample_rate: int,
    seconds: float,
    count: int = 1,
) -> list[np.ndarray]:
    """The `count` best non-overlapping windows of `seconds` each.

    "Best" is the window whose frames are most consistently voiced: the score is
    the fraction of frames above a fraction of the clip's own RMS, tie-broken on
    mean energy. That prefers continuous speech over a stretch that is half
    pause, and it penalises a window that is loud only because it clipped.

    This is what lets a 30-60 s reference feed an engine capped at five seconds
    without throwing the rest away arbitrarily: the caller can take several
    windows and compare what each one produces.
    """
    width = int(seconds * sample_rate)
    if width <= 0 or samples.size <= width:
        return [samples]

    window = max(1, int(0.02 * sample_rate))
    energies = _frame_rms(samples, window)
    if energies.size == 0:
        return [samples[:width]]

    rms = float(np.sqrt((samples.astype(np.float64) ** 2).mean())) or 1.0
    voiced = (energies > 0.15 * rms).astype(np.float64)
    frames_per_window = max(1, width // window)

    voiced_sum = np.concatenate([[0.0], np.cumsum(voiced)])
    energy_sum = np.concatenate([[0.0], np.cumsum(energies)])
    last_start = energies.size - frames_per_window
    if last_start < 0:
        return [samples[:width]]

    starts = np.arange(last_start + 1)
    ends = starts + frames_per_window
    voiced_score = (voiced_sum[ends] - voiced_sum[starts]) / frames_per_window
    energy_score = (energy_sum[ends] - energy_sum[starts]) / frames_per_window
    ranking = np.lexsort((-energy_score, -voiced_score))

    chosen: list[np.ndarray] = []
    taken: list[int] = []
    for frame_start in ranking:
        if len(chosen) >= count:
            break
        if any(abs(int(frame_start) - other) < frames_per_window for other in taken):
            continue
        taken.append(int(frame_start))
        offset = int(frame_start) * window
        chosen.append(samples[offset : offset + width])
    return chosen or [samples[:width]]


@dataclass(frozen=True)
class SpeakerReference:
    """A long reference built from one speaker's recordings."""

    samples: np.ndarray
    sample_rate: int
    sources: tuple[str, ...]

    source_paths: tuple[Path, ...] = ()

    @property
    def duration_seconds(self) -> float:
        return self.samples.size / self.sample_rate

    @property
    def meets_protocol(self) -> bool:
        """Whether this reaches the 30 s an evaluation protocol asks for."""
        return self.duration_seconds >= REFERENCE_MIN_SEC


def build_speaker_reference(
    paths: list[Path] | tuple[Path, ...],
    target_seconds: float = REFERENCE_TARGET_SEC,
    gap_ms: float = 200.0,
    labels: list[str] | tuple[str, ...] | None = None,
) -> SpeakerReference:
    """Concatenate one speaker's recordings into a single long reference.

    The corpus ships 3-6 s read utterances, so no single file is a 30-60 s
    reference; several from the same speaker are. Each is silence-trimmed before
    joining and separated by a short gap, so the result is continuous speech by
    one voice rather than a sequence padded with the original lead-ins.

    Files are consumed in order and the loop stops once the target is reached, so
    a reference is reproducible for a given speaker rather than depending on how
    many recordings that speaker happened to submit.
    """
    if not paths:
        raise ValueError("No recordings to build a reference from.")

    rate: int | None = None
    pieces: list[np.ndarray] = []
    used: list[str] = []
    used_paths: list[Path] = []
    total = 0.0

    for index, path in enumerate(paths):
        samples, sample_rate = read_audio(path)
        if rate is None:
            rate = sample_rate
        elif sample_rate != rate:
            samples = resample_mono(samples, sample_rate, rate)

        samples = trim_silence(samples, rate)
        if samples.size == 0:
            continue

        if pieces:
            pieces.append(np.zeros(int(gap_ms / 1000.0 * rate), dtype=np.float32))
        pieces.append(samples)
        used.append(labels[index] if labels is not None else Path(path).name)
        used_paths.append(Path(path))
        total += samples.size / rate
        if total >= target_seconds:
            break

    if not pieces or rate is None:
        raise ValueError("Every recording was silent.")

    return SpeakerReference(np.concatenate(pieces), rate, tuple(used), tuple(used_paths))


def prepare_voice_prompt(
    source: str | Path | SpeakerReference,
    target_sample_rate: float | int,
    max_seconds: float = MAX_VOICE_PROMPT_SEC,
    window_index: int = 0,
    window_count: int = 1,
) -> tuple[np.ndarray, float, float]:
    """Trim a speaker sample to the supported reference window.

    Returns mono samples at `target_sample_rate`, the source duration and the
    duration used, so a caller can tell how much of a long reference was kept.

    When the source is longer than the cap, the kept seconds are taken from the
    best-voiced window rather than the opening, because a reading often starts
    with a breath or a hesitation. `window_index` selects among the top
    `window_count` windows.
    """
    if isinstance(source, SpeakerReference):
        samples, sample_rate = source.samples, source.sample_rate
    else:
        samples, sample_rate = read_audio(source)

    original_seconds = samples.size / sample_rate
    samples = resample_mono(samples, sample_rate, int(target_sample_rate))
    rate = int(target_sample_rate)
    samples = trim_silence(samples, rate)

    if samples.size == 0:
        raise ValueError("The sample contains no detectable speech.")

    if samples.size > int(max_seconds * rate):
        windows = best_windows(samples, rate, max_seconds, count=max(window_count, window_index + 1))
        samples = windows[min(window_index, len(windows) - 1)]

    return (
        np.ascontiguousarray(samples, dtype=np.float32),
        original_seconds,
        samples.size / rate,
    )

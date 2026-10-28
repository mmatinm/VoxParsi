"""Speech-to-text experiment utilities."""

from __future__ import annotations

import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from voxparsi import audio as audio_io
from voxparsi import correct as llm_correct
from voxparsi import metrics, stt
from voxparsi.dataset import Utterance

Progress = Callable[[int, int, str], None]


def _report(progress: Progress | None, done: int, total: int, label: str) -> None:
    if progress is not None:
        progress(done, total, label)


@dataclass
class STTRow:
    label: str
    speaker: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    seconds: float
    corrected: str | None = None
    corrected_wer: float | None = None
    corrected_cer: float | None = None
    correction_error: str | None = None


@dataclass
class STTExperiment:
    model: str
    device: str
    normalized: bool
    rows: list[STTRow]
    corpus_wer: float
    corpus_cer: float
    mean_wer: float
    median_wer: float
    elapsed_seconds: float
    corrected_corpus_wer: float | None = None
    corrected_corpus_cer: float | None = None
    correction_model: str | None = None
    corrections_applied: int = 0
    correction_failures: int = 0

    @property
    def correction_delta(self) -> float | None:
        if self.corrected_corpus_wer is None:
            return None
        return self.corpus_wer - self.corrected_corpus_wer

    def per_speaker(self) -> list[tuple[str, int, float, float]]:
        grouped: dict[str, list[STTRow]] = {}
        for row in self.rows:
            grouped.setdefault(row.speaker, []).append(row)

        summary = []
        for speaker, rows in grouped.items():
            summary.append(
                (
                    speaker,
                    len(rows),
                    statistics.fmean(r.wer for r in rows),
                    statistics.fmean(r.cer for r in rows),
                )
            )
        return sorted(summary, key=lambda item: -item[2])


def run_stt(
    model: stt.Transcriber,
    utterances: list[Utterance] | tuple[Utterance, ...],
    options: stt.TranscriptionOptions | None = None,
    *,
    normalize: bool = True,
    use_llm_correction: bool = False,
    correction_model: str = llm_correct.DEFAULT_MODEL,
    correction_api_key: str | None = None,
    long_reference_seconds: float | None = None,
    progress: Progress | None = None,
) -> STTExperiment:
    options = options or stt.TranscriptionOptions()
    if long_reference_seconds is not None:
        grouped: dict[str, list[Utterance]] = {}
        for utterance in utterances:
            grouped.setdefault(utterance.speaker, []).append(utterance)

        long_utterances: list[Utterance] = []
        for speaker in sorted(grouped):
            speaker_utterances = sorted(
                grouped[speaker], key=lambda u: (u.utterance_id.lower(), u.wav_path.name)
            )
            reference = audio_io.build_speaker_reference(
                [u.wav_path for u in speaker_utterances],
                target_seconds=long_reference_seconds,
                labels=[u.utterance_id for u in speaker_utterances],
            )
            path = Path(tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name)
            path.write_bytes(audio_io.to_wav_bytes(reference.samples, reference.sample_rate))
            text_by_id = {u.utterance_id: u.text for u in speaker_utterances}
            long_utterances.append(
                Utterance(
                    speaker=speaker,
                    utterance_id=f"reference-{int(reference.duration_seconds)}s",
                    wav_path=path,
                    text=" ".join(text_by_id[label] for label in reference.sources),
                )
            )
        utterances = long_utterances

    started = time.perf_counter()
    rows: list[STTRow] = []
    word_scores: list[metrics.Score] = []
    char_scores: list[metrics.Score] = []
    corrected_word: list[metrics.Score] = []
    corrected_char: list[metrics.Score] = []
    applied = failures = 0

    total = len(utterances)
    for index, utterance in enumerate(utterances, start=1):
        _report(progress, index - 1, total, utterance.label)
        clip_started = time.perf_counter()
        result = stt.transcribe(model, utterance.wav_path, options)
        elapsed = time.perf_counter() - clip_started

        word = metrics.wer(utterance.text, result.text, normalize=normalize)
        char = metrics.cer(utterance.text, result.text, normalize=normalize)
        word_scores.append(word)
        char_scores.append(char)

        row = STTRow(
            label=utterance.label,
            speaker=utterance.speaker,
            reference=utterance.text,
            hypothesis=result.text,
            wer=word.rate,
            cer=char.rate,
            seconds=round(elapsed, 2),
        )

        if use_llm_correction:
            try:
                correction = llm_correct.correct(
                    result.text,
                    model=correction_model,
                    api_key=correction_api_key,
                )
                row.corrected = correction.corrected
                fixed_word = metrics.wer(utterance.text, correction.corrected, normalize=normalize)
                fixed_char = metrics.cer(utterance.text, correction.corrected, normalize=normalize)
                row.corrected_wer = fixed_word.rate
                row.corrected_cer = fixed_char.rate
                corrected_word.append(fixed_word)
                corrected_char.append(fixed_char)
                applied += 1
            except llm_correct.CorrectionUnavailable as error:
                row.correction_error = str(error)
                corrected_word.append(word)
                corrected_char.append(char)
                failures += 1

        rows.append(row)
        _report(progress, index, total, utterance.label)

    return STTExperiment(
        model=model.spec.key,
        device=model.device,
        normalized=normalize,
        rows=rows,
        corpus_wer=metrics.corpus_rate(word_scores),
        corpus_cer=metrics.corpus_rate(char_scores),
        mean_wer=statistics.fmean(s.rate for s in word_scores) if word_scores else 0.0,
        median_wer=statistics.median(s.rate for s in word_scores) if word_scores else 0.0,
        elapsed_seconds=round(time.perf_counter() - started, 1),
        corrected_corpus_wer=metrics.corpus_rate(corrected_word) if corrected_word else None,
        corrected_corpus_cer=metrics.corpus_rate(corrected_char) if corrected_char else None,
        correction_model=correction_model if use_llm_correction else None,
        corrections_applied=applied,
        correction_failures=failures,
    )

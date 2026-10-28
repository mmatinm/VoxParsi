"""Report generation for the STT workflow."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timezone

from voxparsi.experiment import STTExperiment


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def stt_csv(experiment: STTExperiment) -> str:
    buffer = io.StringIO()
    include_correction = experiment.corrected_corpus_wer is not None
    columns = ["id", "speaker", "wer", "cer", "seconds", "reference", "hypothesis"]
    if include_correction:
        columns += ["corrected", "corrected_wer", "corrected_cer", "correction_error"]

    writer = csv.writer(buffer)
    writer.writerow(columns)
    for row in experiment.rows:
        values = [
            row.label,
            row.speaker,
            f"{row.wer:.4f}",
            f"{row.cer:.4f}",
            row.seconds,
            row.reference,
            row.hypothesis,
        ]
        if include_correction:
            values += [
                row.corrected or "",
                "" if row.corrected_wer is None else f"{row.corrected_wer:.4f}",
                "" if row.corrected_cer is None else f"{row.corrected_cer:.4f}",
                row.correction_error or "",
            ]
        writer.writerow(values)
    return buffer.getvalue()


def to_json(stt_run: STTExperiment | None) -> str:
    payload: dict[str, object] = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if stt_run is None:
        return json.dumps(payload, ensure_ascii=False, indent=2)

    payload["speech_to_text"] = {
        "model": stt_run.model,
        "device": stt_run.device,
        "normalized_scoring": stt_run.normalized,
        "utterances": len(stt_run.rows),
        "corpus_wer": stt_run.corpus_wer,
        "corpus_cer": stt_run.corpus_cer,
        "mean_wer": stt_run.mean_wer,
        "median_wer": stt_run.median_wer,
        "elapsed_seconds": stt_run.elapsed_seconds,
        "llm_correction": None
        if stt_run.corrected_corpus_wer is None
        else {
            "model": stt_run.correction_model,
            "corpus_wer": stt_run.corrected_corpus_wer,
            "corpus_cer": stt_run.corrected_corpus_cer,
            "wer_improvement": stt_run.correction_delta,
            "applied": stt_run.corrections_applied,
            "failures": stt_run.correction_failures,
        },
        "rows": [
            {
                "id": row.label,
                "speaker": row.speaker,
                "reference": row.reference,
                "hypothesis": row.hypothesis,
                "wer": row.wer,
                "cer": row.cer,
                "corrected": row.corrected,
                "corrected_wer": row.corrected_wer,
            }
            for row in stt_run.rows
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def markdown(stt_run: STTExperiment | None) -> str:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# VoxParsi performance report",
        "",
        f"Generated {when} · corpus: VoxForge Persian (`voxforge-fa`)",
        "",
    ]

    if stt_run is None:
        return "\n".join(lines)

    lines += [
        "## Speech-to-text",
        "",
        f"- Model: **{stt_run.model}** on `{stt_run.device}`",
        f"- Utterances scored: **{len(stt_run.rows)}**",
        f"- Scoring: {'normalised' if stt_run.normalized else 'raw, unnormalised'}",
        "",
        "| metric | value |",
        "|---|---|",
        f"| Corpus WER | **{_percent(stt_run.corpus_wer)}** |",
        f"| Corpus CER | **{_percent(stt_run.corpus_cer)}** |",
        f"| Mean per-utterance WER | {_percent(stt_run.mean_wer)} |",
        f"| Median per-utterance WER | {_percent(stt_run.median_wer)} |",
        f"| Wall clock | {stt_run.elapsed_seconds}s |",
        "",
    ]

    if stt_run.corrected_corpus_wer is not None:
        delta = stt_run.correction_delta or 0.0
        verdict = (
            f"improved corpus WER by {delta:.1%}"
            if delta > 0
            else f"made corpus WER worse by {abs(delta):.1%}"
        )
        lines += [
            "### LLM post-correction",
            "",
            f"Correction model: `{stt_run.correction_model}` · {verdict}",
            "",
        ]

    return "\n".join(lines)


def bundle(stt_run: STTExperiment | None, _tts_run: object | None = None) -> bytes:
    archive = zipfile.ZipFile(io.BytesIO(), mode="w")
    archive.writestr("report.md", markdown(stt_run))
    archive.writestr("report.json", to_json(stt_run))
    if stt_run is not None:
        archive.writestr("speech_to_text.csv", stt_csv(stt_run))
    data = archive.fp.getvalue()
    archive.close()
    return data


def tts_csv(_experiment: object) -> str:
    raise RuntimeError("Voice cloning is disabled in this project.")

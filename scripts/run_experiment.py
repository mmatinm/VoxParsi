"""Run the corpus STT experiment from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxparsi import dataset, experiment, report, stt  # noqa: E402


def _progress(done: int, total: int, label: str) -> None:
    end = "\n" if done == total else "\r"
    print(f"  [{done:>3}/{total}] {label:<40}", end=end, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the speech-to-text corpus experiment.")
    parser.add_argument("--stt", action="store_true", help="Run the speech-to-text experiment")
    parser.add_argument("--model", default=stt.DEFAULT_MODEL, help=f"one of {sorted(stt.CATALOGUE_BY_KEY)}")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N utterances")
    parser.add_argument("--speaker", action="append", default=[], help="Repeatable speaker filter")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--initial-prompt", default="")
    parser.add_argument("--raw", action="store_true", help="Score without normalising")
    parser.add_argument("--correct", action="store_true", help="Also score with LLM correction")
    parser.add_argument("--correct-model", default='nvidia/nemotron-3-ultra-550b-a55b:free')
    parser.add_argument("--long-reference-seconds", type=float, default=None, help="Use a 30–60 s speaker reference")
    parser.add_argument("--out", type=Path, default=Path("results/experiment"))
    args = parser.parse_args()

    if not args.stt:
        parser.error("Nothing to do: pass --stt.")

    if args.correct:
        from voxparsi import correct as llm_correct
        if not llm_correct.is_configured():
            parser.error(f"--correct needs {llm_correct.API_KEY_ENV} in the environment.")

    utterances = list(dataset.load_corpus())
    if not utterances:
        print("voxforge-fa/ has no usable utterances.", file=sys.stderr)
        return 1
    if args.speaker:
        wanted = set(args.speaker)
        utterances = [u for u in utterances if u.speaker in wanted]
        if not utterances:
            print(f"No utterances for {sorted(wanted)}.", file=sys.stderr)
            return 1

    model = stt.load_model(args.model, args.device)
    options = stt.TranscriptionOptions(
        beam_size=args.beam_size,
        temperature=args.temperature,
        initial_prompt=args.initial_prompt or None,
    )
    selected = utterances[: args.limit] if args.limit else utterances
    run = experiment.run_stt(
        model,
        selected,
        options,
        normalize=not args.raw,
        use_llm_correction=args.correct,
        correction_model=args.correct_model,
        long_reference_seconds=args.long_reference_seconds,
        progress=_progress,
    )

    print(f"\nSpeech-to-text over {len(selected)} utterances:")
    print(f"  corpus WER {run.corpus_wer:.2%}   corpus CER {run.corpus_cer:.2%}")
    print(f"  median WER {run.median_wer:.2%}   {run.elapsed_seconds}s")
    if run.correction_delta is not None:
        print(f"  after correction: WER {run.corrected_corpus_wer:.2%} ({run.correction_delta:+.2%})")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(report.markdown(run), encoding="utf-8")
    (args.out / "report.json").write_text(report.to_json(run), encoding="utf-8")
    (args.out / "speech_to_text.csv").write_text(report.stt_csv(run), encoding="utf-8")
    print(f"\nWrote {args.out}/report.md and friends")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

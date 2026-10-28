"""Reader for the VoxForge Persian corpus in `voxforge-fa/`.

Layout is one directory per submission:

    <speaker>/etc/PROMPTS           "FA-<speaker>/mfc/<utt> <persian text>"
    <speaker>/etc/prompts-original  "<utt> <persian text>"
    <speaker>/wav/<utt>.wav         48 kHz mono 16-bit PCM

`PROMPTS` is the canonical transcript file; `prompts-original` is read only
where a submission is missing it. Utterances whose prompt has no matching wav
(or the reverse) are skipped rather than reported as empty, since several
submissions ship prompts for recordings that were never uploaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "voxforge-fa"


@dataclass(frozen=True)
class Utterance:
    """One recording and the text the speaker was asked to read."""

    speaker: str
    utterance_id: str
    wav_path: Path
    text: str

    @property
    def label(self) -> str:
        """Identifier for pickers and evaluation reports."""
        return f"{self.speaker}/{self.utterance_id}"


def _parse_prompts(path: Path) -> dict[str, str]:
    """Map utterance id -> transcript from a PROMPTS-style file.

    Both formats put the id in the first whitespace-delimited field; in
    `PROMPTS` that field is a path whose last component is the id.
    """
    prompts: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        head, _, text = line.strip().partition(" ")
        text = text.strip()
        if not head or not text:
            continue
        prompts[head.rsplit("/", 1)[-1]] = text
    return prompts


def _load_speaker(speaker_dir: Path) -> list[Utterance]:
    for name in ("PROMPTS", "prompts-original"):
        prompt_file = speaker_dir / "etc" / name
        if prompt_file.is_file():
            prompts = _parse_prompts(prompt_file)
            break
    else:
        return []

    utterances = []
    for utterance_id, text in sorted(prompts.items()):
        wav_path = speaker_dir / "wav" / f"{utterance_id}.wav"
        if wav_path.is_file():
            utterances.append(Utterance(speaker_dir.name, utterance_id, wav_path, text))
    return utterances


@lru_cache(maxsize=1)
def load_corpus(dataset_dir: Path = DATASET_DIR) -> tuple[Utterance, ...]:
    """Every utterance in the corpus, ordered by speaker then utterance id.

    Cached because the Streamlit app rebuilds its picker on every rerun. The
    result is a tuple so the cache cannot hand out a mutable list.
    """
    if not dataset_dir.is_dir():
        return ()
    utterances: list[Utterance] = []
    for speaker_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        utterances.extend(_load_speaker(speaker_dir))
    return tuple(utterances)


def speakers(dataset_dir: Path = DATASET_DIR) -> list[str]:
    """Speaker directory names that yielded at least one usable utterance."""
    return sorted({u.speaker for u in load_corpus(dataset_dir)})


def by_speaker(dataset_dir: Path = DATASET_DIR) -> dict[str, list[Utterance]]:
    """Utterances grouped by speaker, each list in corpus order.

    Voice cloning needs a speaker's recordings together, because no single
    3-6 s utterance is a usable reference on its own.
    """
    grouped: dict[str, list[Utterance]] = {}
    for utterance in load_corpus(dataset_dir):
        grouped.setdefault(utterance.speaker, []).append(utterance)
    return grouped

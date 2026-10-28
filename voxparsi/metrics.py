"""Word- and character-error rate for Persian ASR output.

Persian is written with several codepoints that render identically but are
distinct characters (Arabic yeh vs. Farsi yeh, Arabic kaf vs. keheh), and
Whisper picks whichever its training data favoured. Scoring the raw strings
therefore charges an error for a difference nobody can see or hear, so both
sides are folded onto one spelling first. `normalize=False` on the scoring
functions reports the raw numbers for comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ZWNJ = "‌"

_FOLD = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "ؤ": "و",
    }
)

_DIACRITICS = re.compile(r"[ً-ْٰـٓ-ٕ]")

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹"
                        "٠١٢٣٤٥٦٧٨٩",
                        "01234567890123456789")

_PUNCT = re.compile(r"[.,;:!?،؛؟۔«»\"'()\[\]{}‐-―/\\]")


def normalize_text(text: str) -> str:
    """Fold `text` onto the one spelling both sides are scored in."""
    text = text.translate(_FOLD).translate(_DIGITS)
    text = _DIACRITICS.sub("", text)
    text = _PUNCT.sub(" ", text)
    text = text.replace(ZWNJ, "")
    return " ".join(text.split())


@dataclass(frozen=True)
class Score:
    """One error-rate measurement, with the edits that produced it."""

    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def hits(self) -> int:
        return self.reference_length - self.substitutions - self.deletions

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """Errors per reference token. Empty reference scores 0.0 or 1.0."""
        if self.reference_length == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return self.errors / self.reference_length


def _align(reference: list[str], hypothesis: list[str]) -> Score:
    """Levenshtein distance, counting each edit type separately.

    Full DP table rather than the two-row trick, because the edit counts need a
    backtrace. Inputs here are single utterances, so the table is small.
    """
    n, m = len(reference), len(hypothesis)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                cost[i][j] = 1 + min(
                    cost[i - 1][j - 1],
                    cost[i - 1][j],
                    cost[i][j - 1],
                )

    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] and cost[i][j] == cost[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return Score(subs, dels, ins, n)


def wer(reference: str, hypothesis: str, *, normalize: bool = True) -> Score:
    """Word error rate between a reference transcript and ASR output."""
    if normalize:
        reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    return _align(reference.split(), hypothesis.split())


def cer(reference: str, hypothesis: str, *, normalize: bool = True) -> Score:
    """Character error rate, ignoring word spacing.

    Spaces are dropped rather than scored: Persian compounds are written with a
    space, a ZWNJ or nothing depending on the writer, which would otherwise
    dominate the character count.
    """
    if normalize:
        reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    return _align(list(reference.replace(" ", "")), list(hypothesis.replace(" ", "")))


def corpus_rate(scores: list[Score]) -> float:
    """Pool edits across utterances, rather than averaging per-utterance rates.

    A one-word utterance transcribed wrongly scores 1.0 and would otherwise
    outweigh a twenty-word utterance with one error.
    """
    total_errors = sum(s.errors for s in scores)
    total_reference = sum(s.reference_length for s in scores)
    if total_reference == 0:
        return 0.0 if total_errors == 0 else 1.0
    return total_errors / total_reference

"""Optional LLM post-correction of Whisper output via OpenRouter."""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
API_KEY_ENV = "OPENROUTER_API_KEY"

SYSTEM_PROMPT = """\
You are proofreading Persian (Farsi) speech-recognition output.

Fix only clear acoustic mistakes. Keep wording, order, and meaning stable.
Return only the corrected Persian text with no explanation, no quotes, and no labels.
"""


class CorrectionUnavailable(RuntimeError):
    """Raised when correction cannot run."""


def is_configured(api_key: str | None = None) -> bool:
    key = (api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")).strip()
    return bool(key)


@dataclass(frozen=True)
class Correction:
    original: str
    corrected: str
    model: str

    @property
    def changed(self) -> bool:
        return self.original.strip() != self.corrected.strip()


def correct(
    text: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 20.0,
    api_key: str | None = None,
) -> Correction:
    """Send one transcript to OpenRouter and return the corrected text."""
    key = (api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")).strip()
    if not key:
        raise CorrectionUnavailable(f"{API_KEY_ENV} is not set. Export it to enable LLM post-correction.")
    if not text.strip():
        return Correction(text, text, model)

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text.strip()},
                ],
            },
            timeout=timeout,
        )
    except requests.RequestException as error:
        raise CorrectionUnavailable(f"OpenRouter request failed: {error}") from error

    if response.status_code != 200:
        raise CorrectionUnavailable(
            f"OpenRouter returned {response.status_code}: {response.text[:200]}"
        )

    try:
        payload = response.json()
        corrected = payload["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, IndexError) as error:
        raise CorrectionUnavailable(f"Unexpected OpenRouter response: {error}") from error

    return Correction(text, corrected or text, model)

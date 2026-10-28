"""Voice cloning is disabled in this project."""

from __future__ import annotations


def load_model(*args, **kwargs):
    raise RuntimeError("Voice cloning is disabled in this project.")

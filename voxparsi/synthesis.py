"""Voice cloning is disabled for this project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineSpec:
    key: str
    label: str
    reference_seconds_used: str
    note: str
    download_mb: int
    gated: bool = False


ENGINES: tuple[EngineSpec, ...] = ()
ENGINES_BY_KEY: dict[str, EngineSpec] = {}


def load_engine(_key: str, _device: str = 'cpu'):
    raise RuntimeError('Voice cloning is disabled in this project.')

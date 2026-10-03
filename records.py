"""
records.py — where a song's small bookkeeping files live.

Each artist folder has a hidden .drum_sampler/ folder holding, per song:
  <song>.beats.json    tempo and bar grid
  <song>.parts.json    which DrumSep parts were kept
  <song>.sliced.json   the song is fully cut and its stems were deleted
so the visible library is only samples. Older libraries kept these next to the drum stem in
drum_stems/; they're moved over the first time they're looked up.
"""
from __future__ import annotations

from pathlib import Path


def record(stem: Path, kind: str) -> Path:
    """Path of the <kind> record for the song whose drum stem is (or was) `stem`
    (<Artist>/drum_stems/<song>.flac)."""
    new = stem.parent.parent / ".drum_sampler" / f"{stem.stem}.{kind}.json"
    old = stem.with_suffix(f".{kind}.json")
    if old.exists() and not new.exists():
        new.parent.mkdir(parents=True, exist_ok=True)
        old.replace(new)
        try:
            old.parent.rmdir()  # drum_stems/ goes once it's empty
        except OSError:
            pass
    return new

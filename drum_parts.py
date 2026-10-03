"""
drum_parts.py — split a drum stem into its parts with MDX23C DrumSep (by jarredou & aufr33):
kick, snare, toms, hi-hat, ride, crash. Runs on this Mac (Apple GPU), ~1 min per 3.5-min song.

Compared on Bonobo / deadmau5 / ROSALÍA against LarsNet, it found real snares and hats where
LarsNet put almost everything in the kick, and lost far less of the sound (-26 dB vs -9 dB
on live drums).

Parts are saved as FLAC (lossless, about half the size of WAV) and parts that are essentially
silent in a song (e.g. no ride) are not kept. Which parts were kept is recorded in
<Artist>/.drum_sampler/<song>.parts.json, so a song is never split twice.
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
MODEL_DIR = Path("~/.cache/drum_sampler/models").expanduser()
PARTS = {"kick": "kick", "snare": "snare", "toms": "toms", "hh": "hihat", "ride": "ride", "crash": "crash"}
SILENT_DB = -40.0  # a part this far below the full drums is treated as absent

_separator = None


def _get_separator(out_dir: Path):
    global _separator
    from audio_separator.separator import Separator

    if _separator is None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _separator = Separator(log_level=logging.WARNING, model_file_dir=str(MODEL_DIR),
                               output_dir=str(out_dir), output_format="FLAC")
        _separator.load_model(MODEL)
    _separator.output_dir = str(out_dir)
    if getattr(_separator, "model_instance", None) is not None:
        _separator.model_instance.output_dir = str(out_dir)
    return _separator


def record_path(drum_stem: Path) -> Path:
    from records import record

    return record(drum_stem, "parts")


def part_stem(parts_dir: Path, part: str, song: str) -> Path:
    return parts_dir / part / f"{song}.flac"


def kept_parts(drum_stem: Path) -> list[str] | None:
    """Parts already split for this song, or None if it hasn't been split yet."""
    p = record_path(drum_stem)
    return json.loads(p.read_text())["kept"] if p.exists() else None


def split(drum_stem: Path, parts_dir: Path, song: str) -> list[str]:
    """Split one drum stem into <parts_dir>/<part>/<song>.flac; returns the parts kept."""
    with tempfile.TemporaryDirectory(prefix="drumsep_") as tmp:
        sep = _get_separator(Path(tmp))
        # plain names: the tool rewrites separators like "__" in custom names
        names = {k: f"part{v}" for k, v in PARTS.items()}
        try:
            files = sep.separate(str(drum_stem), custom_output_names=names)
        except TypeError:  # older audio-separator without custom names
            files = sep.separate(str(drum_stem))
        drums, _ = sf.read(drum_stem, always_2d=True)
        ref = np.sqrt(np.mean(drums ** 2)) + 1e-12
        kept = []
        for f in files:
            f = Path(tmp) / Path(f).name
            key = next((k for k in PARTS if f.stem == f"part{PARTS[k]}" or f"({k})" in f.name), None)
            if key is None:
                continue
            audio, _ = sf.read(f, always_2d=True)
            level = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) / ref + 1e-12)
            if level < SILENT_DB:
                continue
            dest = part_stem(parts_dir, PARTS[key], song)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), dest)
            kept.append(PARTS[key])
    record_path(drum_stem).parent.mkdir(parents=True, exist_ok=True)
    record_path(drum_stem).write_text(json.dumps({"kept": sorted(kept), "model": MODEL}))
    return sorted(kept)

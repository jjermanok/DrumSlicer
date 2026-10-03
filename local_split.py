"""
local_split.py — separate drums on this Mac with Demucs (free, no lalal.ai minutes).

htdemucs_ft is Demucs's best model. It's a bag of four models, one per instrument, and the
drum stem comes only from the drum model (weights [1,0,0,0]), so only that one is run: the
drum stem is bit-identical to running all four, ~5x faster (~15 s per song on an M1 Pro).
The no-drums track is the original minus the drums.
On "SubLab - Maera" it measured as clean as lalal.ai (5.5% tonal bleed vs 6.3%).
"""
from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

_models: dict = {}


def _load_bag(model_name: str):
    from demucs.pretrained import get_model

    if model_name not in _models:
        # The model lives on the Hugging Face Hub and Demucs re-checks it on every load (hence
        # the "unauthenticated requests" warning). Use the downloaded copy; only go online the
        # first time, when it isn't downloaded yet.
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            m = get_model(model_name)
        except Exception:
            os.environ.pop("HF_HUB_OFFLINE")
            m = get_model(model_name)
        m.eval()
        _models[model_name] = m
    return _models[model_name]


def _source_model(model, source: str):
    """If this source comes from a single model of a bag (htdemucs_ft: one model per
    instrument), use just that model: identical output, a quarter of the work."""
    from demucs.apply import BagOfModels

    if isinstance(model, BagOfModels):
        d = model.sources.index(source)
        owners = [i for i, w in enumerate(model.weights) if w[d] != 0]
        if len(owners) == 1 and model.weights[owners[0]][d] == 1.0:
            return model.models[owners[0]]
    return model


def separate(path: Path, sources=("drums",), model_name: str = "htdemucs_ft",
             shifts: int = 1) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """({source: audio}, original mix, sample_rate); audio shaped (channels, samples)."""
    import torch
    from demucs.apply import apply_model

    bag = _load_bag(model_name)
    y, _ = librosa.load(path, sr=bag.samplerate, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out = {}
    for source in sources:
        model = _source_model(bag, source)
        with torch.no_grad():
            res = apply_model(model, torch.tensor(y, dtype=torch.float32)[None], device=device,
                              shifts=shifts, split=True, overlap=0.25, progress=False)[0].numpy()
        out[source] = res[model.sources.index(source)][:, :y.shape[-1]]
    n = min([y.shape[-1]] + [a.shape[-1] for a in out.values()])
    return {k: a[:, :n] for k, a in out.items()}, y[:, :n], bag.samplerate


def has_vocals(vocals: np.ndarray, mix: np.ndarray, sr: int, min_seconds: float = 15.0) -> tuple[bool, float]:
    """Instrumentals still give a faint vocal track (bleed), so count the seconds where the
    vocals are within 12 dB of the whole song. Measured: Reliquia 143 s, Bonobo "Cycles"
    (vocal samples) 43 s, instrumentals by John Tejada / deadmau5 0-1 s."""
    hop = sr // 10
    v = librosa.feature.rms(y=vocals.mean(axis=0), frame_length=2 * hop, hop_length=hop)[0]
    m = librosa.feature.rms(y=mix.mean(axis=0), frame_length=2 * hop, hop_length=hop)[0]
    seconds = float(((20 * np.log10((v + 1e-9) / (m + 1e-9))) > -12).sum() * hop / sr)
    return seconds >= min_seconds, seconds


def _write(dest: Path, audio: np.ndarray, sr: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest = dest.with_suffix(".flac")  # lossless, about half the size of WAV
    tmp = dest.with_name(dest.stem + ".part.flac")
    sf.write(tmp, audio.T, sr, format="FLAC", subtype="PCM_24")
    tmp.replace(dest)


def write_stems(path: Path, drums_dest: Path | None, rest_dest: Path | None,
                vocals_dest: Path | None = None, model_name: str = "htdemucs_ft") -> tuple[bool, float] | None:
    """Write whichever stems have a destination. The no-drums track is the original minus the
    drums. Returns (has vocals, seconds of vocals) when vocals were requested; the vocal stem is
    only written if there are vocals."""
    sources = (["drums"] if drums_dest else []) + (["vocals"] if vocals_dest else [])
    stems, mix, sr = separate(path, tuple(sources), model_name)
    if drums_dest:
        _write(drums_dest, stems["drums"], sr)
        if rest_dest:
            _write(rest_dest, mix - stems["drums"], sr)
    if vocals_dest:
        found = has_vocals(stems["vocals"], mix, sr)
        if found[0]:
            _write(vocals_dest, stems["vocals"], sr)
        return found
    return None

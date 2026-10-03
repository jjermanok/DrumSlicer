"""
variation.py — does a song's drum part change, or is it one loop the whole way through?

Runs on your Mac before anything is sent to lalal.ai (costs no lalal minutes):
  1. separate the drums locally with Demucs (~35 s for a 3.5 min song on an M1 Pro).
     Analysing the full mix doesn't work: changes in the other instruments look like
     drum changes.
  2. onset envelopes in three bands (low = kick, mid = snare/clap, high = hats/cymbals)
  3. cut the drums into 10 s windows and compare every pair, sliding one against the other
     to find the best alignment. No tempo or bar detection needed, so tempo mistakes
     (e.g. 100 BPM read as 133) can't break it.
  4. if even the least-alike pairs of windows still match closely, it's one loop

If the drums are basically one loop, only an excerpt needs to go to lalal.ai: the stretch
with the most unusual windows (fills, variations), so they still make it into the pack.
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

SR = 22050
HOP = 256
FPS = SR / HOP
WINDOW_S = 10.0
STEP_S = 5.0
BANDS = ((0, 150), (150, 2500), (2500, SR / 2))

_model = None


@dataclass
class DrumVariation:
    windows: int            # windows with drums playing
    consistency: float      # similarity of the least-alike 10% of window pairs (1.0 = identical)
    starts: np.ndarray      # window start times, seconds
    drums: np.ndarray       # per window: drums playing?
    novelty: np.ndarray     # per window: how unlike the rest of the song's drums it is
    duration: float
    threshold: float = 0.6

    @property
    def repetitive(self) -> bool:
        return self.windows >= 6 and self.consistency >= self.threshold

    def summary(self) -> str:
        if self.windows < 6:
            return f"not enough drums to judge ({self.windows * STEP_S:.0f} s)"
        verdict = "basically one loop" if self.repetitive else "drums change"
        return f"{verdict} (least-alike sections match {self.consistency:.0%}, threshold {self.threshold:.0%})"


def local_drums(path) -> np.ndarray:
    """Mono drum track at SR via Demucs (htdemucs), on the Apple GPU when available."""
    global _model
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if _model is None:
        _model = get_model("htdemucs")
        _model.eval()
    y, _ = librosa.load(path, sr=_model.samplerate, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    with torch.no_grad():
        out = apply_model(_model, torch.tensor(y, dtype=torch.float32)[None], device=device,
                          shifts=0, split=True, overlap=0.1, progress=False)
    drums = out[0, _model.sources.index("drums")].mean(axis=0).numpy()
    return librosa.resample(drums, orig_sr=_model.samplerate, target_sr=SR)


def _band_envelopes(drums: np.ndarray) -> np.ndarray:
    S = np.abs(librosa.stft(drums, n_fft=1024, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
    envs = np.array([librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[(freqs >= lo) & (freqs < hi)] + 1e-9),
                                                  sr=SR, hop_length=HOP) for lo, hi in BANDS])
    return envs / (np.percentile(envs, 99, axis=1, keepdims=True) + 1e-9)


def _aligned_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Best normalised cross-correlation of window a slid across the longer stretch b,
    averaged over the three bands."""
    n = a.shape[1]
    sims = []
    for x, y in zip(a, b):
        x = x - x.mean()
        nx = np.linalg.norm(x) + 1e-9
        corr = np.correlate(y, x, mode="valid")             # sum x * y[k:k+n]
        csum = np.cumsum(np.concatenate([[0.0], y]))
        csq = np.cumsum(np.concatenate([[0.0], y * y]))
        seg_sum = csum[n:] - csum[:-n]
        seg_var = np.maximum(csq[n:] - csq[:-n] - seg_sum ** 2 / n, 0)
        sims.append(corr / (nx * (np.sqrt(seg_var) + 1e-9)))  # x is zero-mean, so y's mean drops out
    return float(np.max(np.mean(sims, axis=0)))


def analyze_drums(drums: np.ndarray, threshold: float = 0.6) -> DrumVariation:
    duration = len(drums) / SR
    envs = _band_envelopes(drums)
    w, step = int(WINDOW_S * FPS), int(STEP_S * FPS)
    starts = np.arange(0, max(1, envs.shape[1] - 2 * w + 1), step)
    rms = np.array([np.sqrt(np.mean(drums[int(s * HOP):int((s + w) * HOP)] ** 2)) for s in starts])
    drum_on = rms > np.percentile(rms, 90) * 10 ** (-18 / 20) if len(rms) else rms.astype(bool)
    idx = np.where(drum_on)[0]
    empty = DrumVariation(len(idx), 0.0, starts / FPS, drum_on, np.zeros(len(starts)), duration, threshold)
    if len(idx) < 6:
        return empty
    sim = np.full((len(starts), len(starts)), np.nan)
    for p, i in enumerate(idx):
        for j in idx[p + 1:]:
            if abs(int(starts[j]) - int(starts[i])) < w:  # overlapping windows trivially match
                continue
            ab = _aligned_similarity(envs[:, starts[i]:starts[i] + w], envs[:, starts[j]:starts[j] + 2 * w])
            ba = _aligned_similarity(envs[:, starts[j]:starts[j] + w], envs[:, starts[i]:starts[i] + 2 * w])
            sim[i, j] = sim[j, i] = max(ab, ba)
    pairs = sim[np.triu_indices(len(starts), 1)]
    pairs = pairs[~np.isnan(pairs)]
    consistency = float(np.percentile(pairs, 10)) if len(pairs) else 1.0
    novelty = np.zeros(len(starts))
    for i in idx:
        row = sim[i][~np.isnan(sim[i])]
        novelty[i] = 1.0 - float(np.median(row)) if len(row) else 0.0
    return DrumVariation(len(idx), consistency, starts / FPS, drum_on, novelty, duration, threshold)


def analyze(path, threshold: float = 0.6) -> DrumVariation:
    return analyze_drums(local_drums(path), threshold)


def best_excerpt(v: DrumVariation, seconds: float) -> tuple[float, float]:
    """(start, length): the stretch with the most drums and the most unusual windows."""
    if len(v.starts) == 0 or v.duration <= seconds:
        return 0.0, v.duration
    n = max(1, int((seconds - WINDOW_S) // STEP_S) + 1)  # windows that fit inside the excerpt
    best, best_i = (-1.0, -1.0), 0
    for i in range(0, max(1, len(v.starts) - n + 1)):
        score = (float(v.drums[i:i + n].sum()), float(v.novelty[i:i + n].sum()))
        if score > best:
            best, best_i = score, i
    start = float(v.starts[best_i])
    return start, min(seconds, v.duration - start)

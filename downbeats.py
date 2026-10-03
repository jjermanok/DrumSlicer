"""
downbeats.py — where do the bars start?

Uses Beat This (CPJKU, 2024), a neural beat/downbeat tracker, on the *full mix*. Downbeats
are mostly heard in the music (chord changes, bass, vocal phrases), not in the drums: on a
test of 33 drum loops with a chord progression underneath, Beat This on the mix found the
right downbeat for 33/33 (forced to 4/4), vs 10/33 for the old kick-based guess on the
drum stem.

The result is saved as <Artist>/.drum_sampler/<song>.beats.json (see records.py); it lines up
with the stem (same start, same length) and doesn't need recomputing.
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np

_tracker = None


def _beat_this(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    global _tracker
    import torch
    from beat_this.inference import Audio2Beats

    if _tracker is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _tracker = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
    beats, downbeats = _tracker(y, sr)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


WINDOW = 16        # beats per local tempo measurement
MAX_BPM = 180.0    # above this, the tracker is counting half-beats: halve
INT_TOL = 0.35     # BPM: round to a whole number unless further off than this
HALF_TOL = 0.15    # BPM: else round to .5 if this close
DRIFT = 0.015      # local tempo spread (p90/p10 - 1) above this = the tempo really drifts


def local_periods(beats: np.ndarray) -> np.ndarray:
    """Beat period measured over many short stretches, skipping stretches where the tracker
    dropped or doubled a beat. A slip in one place can't bend the estimate."""
    ibi = np.diff(beats)
    med = float(np.median(ibi))
    out = []
    for i in range(0, len(beats) - WINDOW, 4):
        seg = beats[i:i + WINDOW]
        if np.all(np.abs(np.diff(seg) - med) < 0.2 * med):
            out.append(np.polyfit(np.arange(WINDOW), seg, 1)[0])
    return np.array(out) if len(out) >= 3 else np.array([med])


def coherence(beats: np.ndarray, bpm: float, factor: int, window: int = 64) -> float:
    """How well a steady grid at `bpm` fits the tracked beats, allowing a different anchor in
    every stretch of `window` beats (so section shifts don't count against it): the typical
    phase concentration (1.0 = every beat exactly on the grid)."""
    period = 60.0 / (bpm * factor)  # tracked beats may be at `factor` x the chosen tempo
    ph = np.exp(2j * np.pi * (beats / period))
    rs = [abs(ph[i:i + window].mean()) for i in range(0, max(1, len(beats) - window), window // 4)]
    return float(np.median(rs))


def choose_tempo(beats: np.ndarray) -> tuple[float, bool, float]:
    """(bpm, steady, raw bpm). Halved above MAX_BPM. Among nearby whole numbers and halves, the
    grid that fits the beats best wins, but a whole number is preferred unless a half fits
    clearly better; if neither fits well, the measured value is kept to 0.01 BPM."""
    periods = local_periods(beats)
    raw = 60.0 / float(np.median(periods))
    bpm, factor = raw, 1
    while bpm > MAX_BPM:
        bpm, factor = bpm / 2, factor * 2
    spread = float(np.percentile(periods, 90) / np.percentile(periods, 10) - 1) if len(periods) >= 5 else 0.0
    steady = spread <= DRIFT
    ints = [float(np.floor(bpm)), float(np.ceil(bpm))]
    halves = [np.floor(bpm) + 0.5]
    exact = round(bpm, 2)
    score = {c: coherence(beats, c, factor) for c in set(ints + halves + [exact])}
    best_int = max(ints, key=score.get)
    best_half = max(halves, key=score.get)
    if score[best_int] >= score[exact] - 0.05 and score[best_int] >= score[best_half] - 0.02:
        return best_int, steady, raw
    if score[best_half] >= score[exact] - 0.05:
        return float(best_half), steady, raw
    return exact, steady, raw


def _circ_mean(phases: np.ndarray) -> float:
    return float(np.angle(np.mean(np.exp(2j * np.pi * phases))) / (2 * np.pi)) % 1.0


def build_grid(beats: np.ndarray, model_db: np.ndarray, duration: float, bpm: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Steady grid at `bpm`, re-anchored wherever the song's beat shifts (e.g. a half-beat slip
    after a breakdown). Returns (grid beats, downbeats, number of sections)."""
    period = 60.0 / bpm
    phase = (beats / period) % 1.0
    # local phase: circular mean over the neighbouring 16 beats
    n = len(beats)
    local = np.array([_circ_mean(phase[max(0, i - 8):i + 8]) for i in range(n)])
    # section boundaries where the local phase moves by > 0.2 of a beat and stays moved
    bounds = [0]
    ref = local[0]
    for i in range(1, n):
        d = (local[i] - ref + 0.5) % 1.0 - 0.5
        if abs(d) > 0.2:
            ahead = local[i:i + 8]
            if len(ahead) >= 4 and np.all(np.abs((ahead - local[i] + 0.5) % 1.0 - 0.5) < 0.1):
                bounds.append(i)
                ref = local[i]
        else:
            ref = _circ_mean(np.array([ref, local[i]]))  # follow slow wobble
    bounds.append(n)
    grid, downs = [], []
    for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        ph = _circ_mean(phase[a:b])
        start = 0.0 if s == 0 else beats[a] - 0.5 * period
        end = duration if b == n else beats[b] - 0.5 * period
        k0 = np.ceil((start / period) - ph)
        seg = (np.arange(k0, (end / period) - ph) + ph) * period
        seg = seg[(seg >= start) & (seg < end)]
        if len(seg) == 0:
            continue
        in_seg = model_db[(model_db >= start) & (model_db < end)]
        votes = np.bincount([int(np.argmin(np.abs(seg - d))) % 4 for d in in_seg], minlength=4) if len(in_seg) else np.array([1, 0, 0, 0])
        p = int(np.argmax(votes))
        grid.append(seg)
        downs.append(seg[p::4])
    return np.concatenate(grid), np.concatenate(downs), len(bounds) - 1


def find_grid(y: np.ndarray, sr: int) -> dict:
    """{'tempo', 'beats', 'downbeats', ...} in seconds, 4 beats per bar.

    Assumes no tempo drift. The tempo is measured over many short stretches (so one tracker
    slip can't bend it), halved if above 180 BPM (the tracker counted half-beats), then set to
    the whole number whose steady grid fits the beats best, unless a .5 fits clearly better
    (e.g. 87.5). The grid is steady at that tempo but re-anchored where the song's beat shifts
    (a half-beat slip after a breakdown); each section's downbeat comes from the model's own
    downbeat votes. Only if the tempo genuinely drifts are the tracked beats kept as they are.
    Tested: 33/33 downbeats on the loop benchmark; Shokoufeh 117, Uncasually 126, Before 87.5."""
    y = librosa.to_mono(y) if y.ndim > 1 else y
    beats, model_db = _beat_this(y, sr)
    if len(beats) < 8:
        return {"tempo": 0.0, "beats": [], "downbeats": []}
    bpm, steady, raw = choose_tempo(beats)
    factor = max(1, int(round(raw / bpm)))
    conf = round(coherence(beats, bpm, factor), 3)
    if not steady:
        votes = np.bincount([int(np.argmin(np.abs(beats - d))) % 4 for d in model_db], minlength=4)
        return {"tempo": bpm, "beats": beats.tolist(), "downbeats": beats[int(votes.argmax())::4].tolist(),
                "steady_grid": False, "measured_tempo": round(raw, 2), "confidence": conf}
    if factor > 1:  # tracker counted half-beats: keep every other beat, the set the downbeats sit on
        hits = [sum(np.min(np.abs(beats[p::factor] - d)) < 0.05 for d in model_db) for p in range(factor)]
        beats = beats[int(np.argmax(hits))::factor]
    grid, downs, sections = build_grid(beats, model_db, len(y) / sr, bpm)
    return {"tempo": bpm, "beats": grid.tolist(), "downbeats": downs.tolist(), "steady_grid": True,
            "sections": sections, "measured_tempo": round(raw, 2), "confidence": conf}


def grid_path(stem: Path) -> Path:
    from records import record

    return record(stem, "beats")


def save_grid(stem: Path, grid: dict, source: str) -> None:
    grid_path(stem).parent.mkdir(parents=True, exist_ok=True)
    grid_path(stem).write_text(json.dumps({**grid, "source": source}))


def load_grid(stem: Path) -> dict | None:
    p = grid_path(stem)
    return json.loads(p.read_text()) if p.exists() else None


def grid_for_file(path: Path, sr: int = 22050) -> dict:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return find_grid(y, sr)

#!/usr/bin/env python3
"""
drum_sampler.py — turn songs into a drum sample pack.

Pipeline, per song:
  1. Upload the mp3 to lalal.ai and extract the drum stem (API v1).
  2. Detect transients in the drum stem.
  3. Export one-shots (single hits, de-duplicated, auto-labelled kick/snare/hat/...)
     and bar-length loops (tempo-detected, de-duplicated) as 24-bit WAVs.

Output layout (artist and title come from the file's tags, else its name):
  <out>/<Artist>/oneshots/kicks/<song>_001.flac   (also snares_claps, hats, cymbals, percussion)
  <out>/<Artist>/loops/<song>_<bpm>bpm_2bar_001.flac
  <out>/<Artist>/drum_stems/<song>.flac           cached, so re-runs don't redo the separation
  <out>/<Artist>/no_drum_stems/<song>.flac        the song with the drums removed
Everything is 24-bit FLAC: lossless (identical to WAV), about half the size, read by Logic and
Ableton. Older .wav files in a library are still recognised.

Usage:
  export LALAL_API_KEY=your_license_key
  python drum_sampler.py song1.mp3 song2.mp3 -o my_pack
  python drum_sampler.py ~/Music/songs/ -o my_pack          # every audio file in a folder
  python drum_sampler.py drums.wav --already-drums -o test  # skip lalal.ai (input is a drum stem)
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path

import librosa
import mutagen
import numpy as np
import requests
import soundfile as sf

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a", ".ogg"}
LALAL_API = "https://www.lalal.ai/api/v1"
HOP = 256


# --------------------------------------------------------------------------- #
# lalal.ai client
# --------------------------------------------------------------------------- #
class LalalError(RuntimeError):
    pass


class Lalal:
    def __init__(self, license_key: str, splitter: str | None = None, extraction: str = "clear_cut"):
        self.s = requests.Session()
        self.s.headers["X-License-Key"] = license_key
        self.splitter = splitter
        self.extraction = extraction

    def _post(self, path: str, **kw) -> dict:
        for attempt in range(5):
            r = self.s.post(f"{LALAL_API}{path}", timeout=600, **kw)
            if r.status_code == 429:  # /check/ is limited to 30 req/min
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise LalalError(f"{path} -> HTTP {r.status_code}: {r.text[:500]}")
            return r.json()
        raise LalalError(f"{path} -> rate limited repeatedly")

    def minutes_left(self) -> float | None:
        try:
            return float(self._post("/limits/minutes_left/")["minutes_left"])
        except Exception:
            return None

    def upload(self, path: Path) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.name)
        with open(path, "rb") as f:
            data = self._post(
                "/upload/",
                data=f,
                headers={
                    "Content-Disposition": f'attachment; filename="{safe_name}"',
                    "Content-Type": "application/octet-stream",
                },
            )
        return data["id"]

    def split_drums(self, source_id: str) -> str:
        presets = {"stem": "drum", "encoder_format": "flac", "extraction_level": self.extraction}
        if self.splitter:
            presets["splitter"] = self.splitter
        data = self._post(
            "/split/stem_separator/",
            json={"source_id": source_id, "presets": presets, "idempotency_key": str(uuid.uuid4())},
        )
        return data["task_id"]

    def check(self, task_ids: list[str]) -> dict:
        return self._post("/check/", json={"task_ids": task_ids})["result"]

    def download(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
        tmp.replace(dest)

    def delete(self, source_id: str) -> None:
        try:
            self._post("/delete/", json={"source_id": source_id})
        except Exception:
            pass  # files expire on their own; not worth failing over


def fmt_time(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def plan_upload(info: Song, args, tmp: Path) -> tuple[Path, bool]:
    """Check locally whether the drums are basically one loop. If so, cut out just the most
    varied stretch so lalal.ai only processes (and bills) that. Returns (file to upload, is_excerpt)."""
    if args.full:
        return info.path, False
    import variation

    log(f"[analyze] {info.path.name}: checking whether the drums change (local, no lalal minutes)...")
    v = variation.analyze(info.path, threshold=args.loop_threshold)
    log(f"[analyze] {info.path.name}: {v.summary()}")
    if not v.repetitive or v.duration <= args.excerpt_seconds * 1.2:
        return info.path, False
    start, length = variation.best_excerpt(v, args.excerpt_seconds)
    clip = tmp / f"{info.name}_excerpt.flac"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
                    "-i", str(info.path), "-c:a", "flac", str(clip)], check=True)
    log(f"[analyze] {info.path.name}: sending only {fmt_time(start)}-{fmt_time(start + length)} to lalal.ai "
        f"(saves ~{(v.duration - length) / 60:.1f} of {v.duration / 60:.1f} min)")
    return clip, True


def save_bar_grid(stem: Path, mix: Path, source: str) -> None:
    """Find bars on the audio actually sent to lalal.ai (same timeline as the stem it returns)."""
    import downbeats

    try:
        stem.parent.mkdir(parents=True, exist_ok=True)
        downbeats.save_grid(stem, downbeats.grid_for_file(mix), source)
    except Exception as e:
        log(f"[beats] {mix.name}: couldn't find bars ({e}); loops will fall back to the drums alone")


def bar_grid_for(info: Song, stem: Path, out: Path) -> dict | None:
    """Saved grid; else the original song if it's the same length as the stem (same timeline);
    else the drum + no-drums stems added back together (for excerpt stems). The original is
    preferred: separated stems don't sum back to the exact mix, and on "Maera" that threw
    Beat This off (100 BPM, no steady grid, vs 137 BPM on the original)."""
    import downbeats

    grid = downbeats.load_grid(stem)
    if grid:
        return grid
    if not stem.exists():
        return None
    try:
        stem_len = librosa.get_duration(path=stem)
        if info.path != stem and abs(librosa.get_duration(path=info.path) - stem_len) < 1.0:
            grid = downbeats.grid_for_file(info.path)
            downbeats.save_grid(stem, grid, "full mix")
            return grid
        d, sr = librosa.load(stem, sr=22050, mono=True)
        base = info.dir(out) / "no_drum_stems"
        for rest_path in (audio_file(base, info.name), audio_file(base, f"{info.name}_excerpt")):
            if rest_path.exists():
                rest, _ = librosa.load(rest_path, sr=sr, mono=True)
                if abs(len(rest) - len(d)) < sr:
                    n = min(len(rest), len(d))
                    grid = downbeats.find_grid(d[:n] + rest[:n], sr)
                    downbeats.save_grid(stem, grid, "drum stem + no-drums stem")
                    return grid
    except Exception as e:
        log(f"[beats] {info.path.name}: couldn't find bars ({e})")
    return None


def vocal_paths(info: Song, vocals_dir: Path) -> tuple[Path, Path]:
    """(acapella, phrases folder) for a song, under the separate vocals library."""
    base = vocals_dir / info.artist
    return audio_file(base / "acapellas", info.name), base / "phrases"


def _no_vocals_registry(vocals_dir: Path) -> Path:
    return vocals_dir / ".no_vocals.txt"


def demucs_stems(songs: list[Song], out: Path, vocals_dir: Path | None = None) -> dict[Path, Path]:
    """Separate each song on this Mac with Demucs (free). Whole song every time, so there's no
    need for the one-loop excerpt check. Existing stems (from either separator) are reused; a
    song that already has drums only gets its vocals separated. Songs found to have no vocals
    are listed in <vocals_dir>/.no_vocals.txt so they aren't checked again."""
    import local_split

    results: dict[Path, Path] = {}
    registry = _no_vocals_registry(vocals_dir) if vocals_dir else None
    no_vocals = set(registry.read_text().splitlines()) if registry and registry.exists() else set()
    for info in songs:
        dest = audio_file(info.dir(out) / "drum_stems", info.name)
        key = f"{info.artist}/{info.name}"
        acapella = vocal_paths(info, vocals_dir)[0] if vocals_dir else None
        need_drums = not dest.exists() and not sliced_marker(dest).exists()
        need_vocals = bool(acapella) and not acapella.exists() and key not in no_vocals
        if not need_drums and not need_vocals:
            log(f"[cache] {info.path.name}: using existing stems")
            results[info.path] = dest
            continue
        what = " and ".join((["drums"] if need_drums else []) + (["vocals"] if need_vocals else []))
        log(f"[demucs] {info.path.name}: separating {what} on this Mac...")
        t0 = time.time()
        try:
            found = local_split.write_stems(info.path, dest if need_drums else None,
                                            info.dir(out) / "no_drum_stems" / f"{info.name}{EXT}",
                                            acapella if need_vocals else None)
        except Exception as e:
            log(f"[error] {info.path.name}: Demucs failed: {e}")
            continue
        note = ""
        if need_vocals and found is not None:
            if found[0]:
                note = f"; vocals found ({found[1]:.0f} s)"
            else:
                note = f"; no vocals ({found[1]:.0f} s above the threshold)"
                registry.parent.mkdir(parents=True, exist_ok=True)
                with open(registry, "a") as f:
                    f.write(key + "\n")
        log(f"[demucs] {info.path.name}: done ({time.time() - t0:.0f} s){note}")
        if need_drums:
            save_bar_grid(dest, info.path, "full mix")
        results[info.path] = dest
    return results


def export_vocal_phrases(acapella: Path, song_slug: str, out_dir: Path, args) -> int:
    """Cut an acapella into phrases at its pauses: 0.5-12 s each, quiet bits dropped, near-
    duplicates (repeated hooks) removed keeping the loudest take."""
    y, sr = librosa.load(acapella, sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    mono = librosa.to_mono(y)
    peak = np.max(np.abs(mono)) + 1e-12

    def pieces(a: int, b: int, top_db: float) -> list[tuple[int, int]]:
        """Non-silent stretches of [a, b); ones too long are re-split with a stricter silence
        threshold, and as a last resort cut at their quietest point."""
        iv = librosa.effects.split(mono[a:b], top_db=top_db, ref=peak, frame_length=2048, hop_length=512) + a
        merged: list[list[int]] = []
        for s0, e0 in iv:  # bridge gaps under 0.3 s (breaths between words)
            if merged and s0 - merged[-1][1] < 0.3 * sr:
                merged[-1][1] = int(e0)
            else:
                merged.append([int(s0), int(e0)])
        res = []
        for s0, e0 in merged:
            if e0 - s0 <= 12 * sr:
                res.append((s0, e0))
            elif top_db > 20:
                res += pieces(s0, e0, top_db - 8)
            else:
                rms = librosa.feature.rms(y=mono[s0:e0], frame_length=2048, hop_length=512)[0]
                lo, hi = len(rms) // 4, 3 * len(rms) // 4
                cut = s0 + (lo + int(np.argmin(rms[lo:hi]))) * 512
                res += pieces(s0, cut, top_db) + pieces(cut, e0, top_db)
        return res

    cands, fps, scores = [], [], []
    for s0, e0 in pieces(0, len(mono), 35.0):
        s0, e0 = max(0, s0 - int(0.02 * sr)), min(len(mono), e0 + int(0.08 * sr))
        seg = mono[s0:e0]
        if len(seg) < 0.5 * sr or db(np.max(np.abs(seg)) / peak) < -24:
            continue
        cands.append((s0, e0))
        fps.append(fingerprint(seg, sr, min(len(seg) / sr, 3.0), 24))
        scores.append(float(np.sqrt(np.mean(seg ** 2))))
    kept = sorted(dedupe(cands, fps, scores, args.vocal_similarity, args.max_vocal_phrases))
    for k, (s0, e0) in enumerate(kept, 1):
        seg = fade(y[..., s0:e0], sr, 5.0, 40.0)
        if args.normalize:
            seg = normalize(seg)
        write_wav(out_dir / f"{song_slug}_phrase_{k:03d}{EXT}", seg, sr)
    return len(kept)


def extract_drum_stems(songs: list[Song], out: Path, client: Lalal, args, poll_s: float = 5.0) -> dict[Path, Path]:
    """Upload + split every song that doesn't already have a cached stem, then poll
    all tasks together and download each drum stem (and the song minus drums) as it finishes.
    Songs whose drums are one loop are cut to an excerpt first (see plan_upload)."""
    results: dict[Path, Path] = {}
    pending: dict[str, tuple[Song, str, Path, bool]] = {}  # task_id -> (song, source_id, dest, excerpt)
    tmp = Path(tempfile.mkdtemp(prefix="drum_sampler_"))

    for info in songs:
        song = info.path
        dest = audio_file(info.dir(out) / "drum_stems", info.name)
        if dest.exists() or sliced_marker(dest).exists():
            log(f"[cache] {song.name}: using existing drum stem")
            results[song] = dest
            continue
        try:
            upload_path, excerpt = plan_upload(info, args, tmp)
            save_bar_grid(dest, upload_path, "full mix" if not excerpt else "full mix (excerpt)")
        except Exception as e:  # analysis is an optimisation; never let it block a song
            log(f"[analyze] {song.name}: analysis failed ({e}); sending the full song")
            upload_path, excerpt = song, False
        log(f"[upload] {song.name}" + (" (excerpt)" if excerpt else ""))
        try:
            source_id = client.upload(upload_path)
            task_id = client.split_drums(source_id)
        except LalalError as e:
            log(f"[error] {song.name}: {e}")
            continue
        pending[task_id] = (info, source_id, dest, excerpt)

    last_progress: dict[str, int] = {}
    while pending:
        time.sleep(poll_s)
        try:
            status = client.check(list(pending))
        except LalalError as e:
            log(f"[warn] status check failed, retrying: {e}")
            continue
        for task_id, info in status.items():
            if task_id not in pending:
                continue
            song_info_, source_id, dest, excerpt = pending[task_id]
            song = song_info_.path
            st = info.get("status")
            if st == "progress":
                p = int(info.get("progress", 0))
                if last_progress.get(task_id) != p:
                    log(f"[split] {song.name}: {p}%" if p else f"[split] {song.name}: queued")
                    last_progress[task_id] = p
            elif st == "success":
                tracks = info["result"]["tracks"]
                stem = next((t for t in tracks if t["type"] == "stem"), None)
                back = next((t for t in tracks if t["type"] == "back"), None)
                if stem is None:
                    log(f"[error] {song.name}: no stem track in result")
                else:
                    client.download(stem["url"], dest)
                    log(f"[done] {song.name}: drum stem saved")
                    results[song] = dest
                if back is not None:
                    suffix = "_excerpt" if excerpt else ""
                    client.download(back["url"], song_info_.dir(out) / "no_drum_stems" / f"{song_info_.name}{suffix}{EXT}")
                    log(f"[done] {song.name}: no-drums track saved")
                client.delete(source_id)
                del pending[task_id]
            elif st in ("error", "cancelled", "server_error"):
                err = info.get("error")
                if isinstance(err, dict):
                    err = err.get("detail")
                log(f"[error] {song.name}: split {st}: {err}")
                del pending[task_id]
    return results


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(msg, flush=True)


def slug(s: str, maxlen: int = 40) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"['\u2019]", "", s)  # "Can't" -> "cant", not "can_t"
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s[:maxlen] or "song"


ONESHOT_FOLDERS = {"kick": "kicks", "snare": "snares_claps", "hat": "hats", "cymbal": "cymbals", "perc": "percussion"}


@dataclass
class Song:
    path: Path
    artist: str  # folder name, human-readable
    name: str    # file-name slug, e.g. "shokoufeh"

    def dir(self, out: Path) -> Path:
        return out / self.artist


def _tag(tags, key: str) -> str:
    v = tags.get(key) if tags else None
    return str(v[0]).strip() if v else ""


def song_info(path: Path) -> Song:
    """Artist/title from the file's tags; falls back to 'Artist - Title' in the file name."""
    stem = re.sub(r"(_(no_)?drums?)?(_split_by_lalalai)?( \(\d+\))?$", "", path.stem, flags=re.I)
    # e.g. "_drum_split_by_lalalai", "_drum_(Split_by_LALAL.AI)"
    splitter_note = r"[\s_(\[-]*(no[\s_])?drums?[\s_(\[]*split[\s_]by[\s_]lalal\.?ai[)\]]*"
    stem = re.sub(r"^\s*\d+([-_. ]+\d+)?[-_. ]+", "", stem)  # leading track numbers: "10 ", "1-06- "
    try:
        tags = mutagen.File(path, easy=True)
    except Exception:
        tags = None
    artist = _tag(tags, "albumartist") or _tag(tags, "artist")
    title = re.sub(splitter_note, "", _tag(tags, "title"), flags=re.I).strip()
    # "Berghain (ft. Björk & Yves Tumor)" -> "Berghain": featured artists make names long, and
    # the artist folder already says whose song it is
    title = re.sub(r"\s*[(\[]?\s*\b(feat|ft|featuring)\b\.?\s[^)\]]*[)\]]?", "", title, flags=re.I).strip()
    if " - " in stem:
        fn_artist, fn_title = stem.split(" - ", 1)
        artist, title = artist or fn_artist, title or fn_title
    artist = re.split(r"\s*,\s*|\s+(?:feat\.?|ft\.?|featuring|&|x)\s+", artist or "Unknown Artist", flags=re.I)[0]
    return Song(path, folder_name(artist), slug(title or stem, maxlen=60))


def folder_name(artist: str) -> str:
    return re.sub(r'[/\\:*?"<>|]+', "_", artist).strip(" .") or "Unknown Artist"


def song_info_for(path: Path, query: str | None, out: Path) -> Song:
    """For a song requested as "Artist - Title" (songs.txt / a playlist), name its folder and
    files after that request rather than the downloaded file's tags, which are often missing
    or wrong ("https_djsoundtop.com", title in the artist field, "Above" for Above & Beyond).
    Stems already made under the tag-based name are moved over instead of redone."""
    info = song_info(path)
    if not query or " - " not in query:
        return info
    artist, title = query.split(" - ", 1)
    title = re.sub(r"\s*[(\[]?\s*\b(feat|ft|featuring)\b\.?\s[^)\]]*[)\]]?", "", title, flags=re.I).strip()
    new = Song(path, folder_name(artist.strip()), slug(title, maxlen=60))
    for sub, suffix in (("drum_stems", ".wav"), ("drum_stems", ".flac"), ("drum_stems", ".beats.json"),
                        ("drum_stems", ".parts.json"), ("drum_stems", ".sliced.json"),
                        (".drum_sampler", ".beats.json"), (".drum_sampler", ".parts.json"),
                        (".drum_sampler", ".sliced.json"),
                        ("no_drum_stems", ".wav"), ("no_drum_stems", ".flac")):
        old_f = info.dir(out) / sub / f"{info.name}{suffix}"
        new_f = new.dir(out) / sub / f"{new.name}{suffix}"
        if old_f.exists() and not new_f.exists() and old_f != new_f:
            new_f.parent.mkdir(parents=True, exist_ok=True)
            old_f.replace(new_f)
    for d in (info.dir(out) / "drum_stems", info.dir(out) / "no_drum_stems", info.dir(out)):
        try:
            d.rmdir()  # only if now empty
        except OSError:
            pass
    return new


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-12))


def fade(seg: np.ndarray, sr: int, fade_in_ms: float, fade_out_ms: float) -> np.ndarray:
    seg = seg.copy()
    n = seg.shape[-1]
    fi = min(n, int(sr * fade_in_ms / 1000))
    fo = min(n, int(sr * fade_out_ms / 1000))
    if fi > 0:
        seg[..., :fi] *= np.linspace(0, 1, fi)
    if fo > 0:
        seg[..., n - fo:] *= np.linspace(1, 0, fo) ** 2
    return seg


def normalize(seg: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    peak = np.max(np.abs(seg))
    return seg if peak < 1e-9 else seg * (10 ** (peak_db / 20) / peak)


EXT = ".flac"
AUDIO_OUT = (".flac", ".wav")  # what a library may contain: new FLAC, older WAV


def audio_file(folder: Path, name: str) -> Path:
    """The existing <folder>/<name>.flac or .wav, else the .flac path to create."""
    for ext in AUDIO_OUT:
        if (folder / f"{name}{ext}").exists():
            return folder / f"{name}{ext}"
    return folder / f"{name}{EXT}"


def audio_glob(folder: Path, pattern: str) -> list[Path]:
    """glob for "<pattern>.<flac|wav>"."""
    return [p for ext in AUDIO_OUT for p in folder.glob(pattern + ext)]


def write_wav(path: Path, seg: np.ndarray, sr: int) -> None:
    """Write a sample: 24-bit FLAC, whatever extension the caller used (lossless, half the size)."""
    path = path.with_suffix(EXT)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, seg.T if seg.ndim == 2 else seg, sr, format="FLAC", subtype="PCM_24")


def fingerprint(mono: np.ndarray, sr: int, dur_s: float, n_frames: int) -> np.ndarray:
    """Time-resolved log-mel fingerprint, z-normalised, for similarity comparison."""
    n = int(sr * dur_s)
    x = np.zeros(n, dtype=np.float32)
    x[: min(n, len(mono))] = mono[:n]
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=32, hop_length=max(1, n // n_frames), n_fft=2048)
    S = librosa.power_to_db(S + 1e-10, ref=1.0)[:, :n_frames]
    v = S.flatten()
    return (v - v.mean()) / (v.std() + 1e-9)


def dedupe(items: list, fps: list[np.ndarray], scores: list[float], thresh: float, cap: int) -> list:
    """Greedy: keep the highest-scoring item, drop anything too similar to an already-kept one."""
    order = np.argsort(scores)[::-1]
    kept, kept_fps = [], []
    for i in order:
        f = fps[i]
        if any(float(np.dot(f, k)) / len(f) > thresh for k in kept_fps):
            continue
        kept.append(items[i])
        kept_fps.append(f)
        if len(kept) >= cap:
            break
    return kept


# --------------------------------------------------------------------------- #
# One-shots
# --------------------------------------------------------------------------- #
@dataclass
class Hit:
    start: int
    end: int
    strength: float
    label: str = "perc"


def classify_hit(mono: np.ndarray, start: int, end: int, sr: int) -> str:
    """Rough spectral heuristic: kick / snare / hat / cymbal / perc.

    Looks only at the energy the hit *adds* (spectrum just after the onset minus the
    spectrum just before it). Otherwise the ringing tail of the previous kick or snare
    swamps quiet hats and they get labelled as whatever they're played over."""
    n = 2048
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1 / sr)

    def power(x: np.ndarray) -> np.ndarray:
        buf = np.zeros(n)
        buf[: len(x)] = x[:n]
        return np.abs(np.fft.rfft(buf * win)) ** 2

    post = power(mono[start:start + n])
    pre = power(mono[max(0, start - n):start])
    new = np.maximum(post - pre, 0)
    total = new.sum() + 1e-12
    band = lambda lo, hi: new[(freqs >= lo) & (freqs < hi)].sum() / total
    low, lowmid, mid, high = band(0, 200), band(200, 800), band(800, 3200), band(3200, sr / 2)
    air = band(9000, sr / 2)

    if low > 0.5:
        return "kick"
    if high > 0.55:
        # cymbals ring on; hats are choked. Compare level ~200 ms in against the attack.
        seg = mono[start:end]
        a, b = int(0.05 * sr), int(0.2 * sr)
        if len(seg) > int(0.3 * sr):
            attack = np.sqrt(np.mean(seg[:a] ** 2)) + 1e-12
            tail = np.sqrt(np.mean(seg[b:b + a] ** 2))
            if db(tail / attack) > -12 or air > 0.45:
                return "cymbal"
        return "hat"
    # crack-heavy snares/claps, or body-heavy snares whose wires add top end (toms don't)
    if (mid > 0.4 and mid > lowmid) or (lowmid > 0.5 and high > 0.08):
        return "snare"
    return "perc"


def find_hits(y: np.ndarray, sr: int, sensitivity: float, max_len_s: float, min_len_s: float,
              floor_db: float) -> tuple[list[Hit], np.ndarray]:
    mono = librosa.to_mono(y)
    env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=HOP, backtrack=True,
                                        units="samples", delta=sensitivity, wait=2)
    onset_frames = librosa.samples_to_frames(onsets, hop_length=HOP)
    peak_all = np.max(np.abs(mono)) + 1e-12
    hits = []
    for i, s in enumerate(onsets):
        nxt = onsets[i + 1] if i + 1 < len(onsets) else len(mono)
        e = min(nxt, s + int(max_len_s * sr))
        seg = mono[s:e]
        if len(seg) < int(min_len_s * sr):
            continue
        pk = np.max(np.abs(seg))
        if db(pk / peak_all) < floor_db:
            continue
        # trim tail where the amplitude envelope has decayed ~48 dB below the hit's peak
        rms = librosa.feature.rms(y=seg, frame_length=512, hop_length=128)[0]
        above = np.where(rms > pk * 10 ** (-48 / 20))[0]
        if len(above):
            e = s + min(len(seg), (above[-1] + 2) * 128)
        if e - s < int(min_len_s * sr):
            continue
        f = min(len(env) - 1, onset_frames[i] + 2)
        strength = float(env[max(0, onset_frames[i] - 1): f + 1].max())
        hits.append(Hit(int(s), int(e), strength))
    return hits, mono


def export_oneshots(y, sr, song_slug, out_dir: Path, args) -> int:
    hits, mono = find_hits(y, sr, args.sensitivity, args.max_oneshot_len, 0.03, args.floor_db)
    if not hits:
        return 0
    for h in hits:
        h.label = classify_hit(mono, h.start, h.end, sr)
    fps = [fingerprint(mono[h.start:h.end], sr, 0.25, 16) for h in hits]
    # De-dupe within each label, then take turns across labels so quiet hats/cymbals
    # still get slots instead of the cap filling up with the loudest kicks and snares.
    per_label: dict[str, list[Hit]] = {}
    for label in {h.label for h in hits}:
        idx = [i for i, h in enumerate(hits) if h.label == label]
        per_label[label] = dedupe([hits[i] for i in idx], [fps[i] for i in idx],
                                  [hits[i].strength for i in idx], args.oneshot_similarity, args.max_oneshots)
    kept: list[Hit] = []
    for rnd in range(args.max_oneshots):
        for label in sorted(per_label):
            if rnd < len(per_label[label]) and len(kept) < args.max_oneshots:
                kept.append(per_label[label][rnd])
    kept.sort(key=lambda h: (h.label, h.start))
    counts: dict[str, int] = {}
    for h in kept:
        counts[h.label] = counts.get(h.label, 0) + 1
        seg = fade(y[..., h.start:h.end], sr, 0.5, min(15.0, (h.end - h.start) / sr * 1000 * 0.3))
        if args.normalize:
            seg = normalize(seg)
        write_wav(out_dir / ONESHOT_FOLDERS[h.label] / f"{song_slug}_{counts[h.label]:03d}{EXT}", seg, sr)
    return len(kept)


# --------------------------------------------------------------------------- #
# Loops
# --------------------------------------------------------------------------- #
def estimate_tempo_and_downbeats(mono: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=HOP)
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=HOP, units="frames")
    if len(beats) < 8:
        return 0.0, np.array([], dtype=int)
    beat_samples = librosa.frames_to_samples(beats, hop_length=HOP)
    # Fit a line through beat positions: averages out frame quantisation, unlike a median interval
    idx = np.arange(len(beat_samples))
    slope = np.polyfit(idx, beat_samples, 1)[0]
    med = float(np.median(np.diff(beat_samples)))
    period = slope if abs(slope - med) / med < 0.05 else med  # fall back if tempo drifts/changes
    tempo = 60.0 * sr / period
    if abs(tempo - round(tempo)) < 0.3:
        tempo = float(round(tempo))
    # Downbeat guess (assumes 4/4): the beat phase where low-end (kick) onsets are strongest
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=HOP))
    low_bins = librosa.fft_frequencies(sr=sr, n_fft=2048) < 150
    low_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[low_bins] + 1e-9), sr=sr)
    scores = [low_env[beats[p::4]].sum() for p in range(4)]
    phase = int(np.argmax(scores))
    return tempo, beat_samples[phase::4]


def export_loops(y, sr, song_slug, out_dir: Path, args, grid: dict | None = None,
                 apple_folder: str | None = None) -> tuple[int, float]:
    """grid: beats/downbeats found on the full mix (see downbeats.py), in seconds on the stem's
    timeline. Without one, falls back to guessing from the drums alone (much less reliable)."""
    mono = librosa.to_mono(y)
    steady = bool(grid and grid.get("steady_grid"))
    if grid and grid.get("tempo", 0) > 0 and grid.get("downbeats"):
        tempo = float(grid["tempo"])
        downbeats = (np.asarray(grid["downbeats"]) * sr).astype(int)
    else:
        tempo, downbeats = estimate_tempo_and_downbeats(mono, sr)
    if tempo <= 0 or len(downbeats) == 0:
        return 0, 0.0
    # Snap each downbeat to the drum hit's true start: the nearest backtracked transient from
    # 20 ms before to 40 ms after (never far enough back to grab a pickup note)
    onsets = librosa.onset.onset_detect(y=mono, sr=sr, hop_length=HOP, backtrack=True, units="samples")
    if len(onsets):
        snapped = []
        for d in downbeats:
            near = onsets[(onsets >= d - int(0.02 * sr)) & (onsets <= d + int(0.04 * sr))]
            snapped.append(int(near[np.argmin(np.abs(near - d))]) if len(near) else int(d))
        downbeats = np.array(snapped)

    bar_len = 4 * 60.0 / tempo * sr
    peak_all = np.max(np.abs(mono)) + 1e-12
    total = 0
    for bars in args.loop_bars:
        cands, fps, scores = [], [], []
        for i in range(0, len(downbeats) - bars, bars):
            s = int(downbeats[i])
            # Steady tempo (the normal case): exactly `bars` bars at the song's BPM from the
            # downbeat, so it's sample-accurate in a DAW. Only when the tempo really drifts, cut
            # to the next detected downbeat instead so the loop still ends on a bar line.
            span = int(downbeats[i + bars]) - s
            local = bars * bar_len / span * tempo
            if steady or abs(local - tempo) < 0.3:
                n, local = int(round(bars * bar_len)), tempo
            elif abs(local - tempo) / tempo > 0.04:  # a missed/extra beat in the grid, not a real bar
                continue
            else:
                n = span
            if s + n > len(mono):
                break
            seg = mono[s:s + n]
            # reject quiet / gappy sections (breakdowns, fills into silence)
            frame_rms = librosa.feature.rms(y=seg, frame_length=2048, hop_length=1024)[0]
            if db(np.sqrt(np.mean(seg ** 2)) / peak_all) < args.loop_floor_db:
                continue
            beat_frames = max(1, int(60.0 / tempo * sr / 1024))
            silent = frame_rms < peak_all * 10 ** (-45 / 20)
            if max_run(silent) >= beat_frames:
                continue
            cands.append((s, n, local))
            fps.append(fingerprint(seg, sr, n / sr, 8 * bars))
            scores.append(float(np.sqrt(np.mean(seg ** 2))))
        kept = sorted(dedupe(cands, fps, scores, args.loop_similarity, args.max_loops))
        for k, (s, n, loop_bpm) in enumerate(kept, 1):
            seg = fade(y[..., s:s + n], sr, 0.5, 3.0)
            if args.normalize:
                seg = normalize(seg)
            bpm = f"{loop_bpm:.2f}".rstrip("0").rstrip(".")  # 126, 87.5, 87.56
            name = f"{song_slug}_{bpm}bpm_{bars}bar_{k:03d}"
            write_wav(out_dir / f"{name}{EXT}", seg, sr)
            if args.apple_loops:
                import apple_loops
                apple_loops.write_apple_loop(apple_loops.USER_LOOPS / (apple_folder or out_dir.parent.name) / f"{name}.caf", seg, sr,
                                             beats=4 * bars, transients=apple_loops.transients_of(librosa.to_mono(seg), sr),
                                             title=name, artist=out_dir.parent.name)
        total += len(kept)
    return total, tempo


def max_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def collect_inputs(paths: list[str]) -> tuple[list[Path], list[str]]:
    """Split inputs into local audio files and 'Artist - Title' song queries."""
    files, queries = [], []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        elif p.is_file():
            files.append(p)
        elif p.suffix.lower() in AUDIO_EXTS or "/" in raw:
            log(f"[warn] not found: {p}")
        else:
            queries.append(raw.strip())
    return files, queries


def remove_from_list(path: Path, query: str) -> None:
    """Drop a finished song's line from the songs file (re-read first, in case it was edited meanwhile)."""
    lines = path.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip() == query:
            del lines[i]
            break
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(lines))
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Turn songs into drum one-shots and loops via lalal.ai.")
    ap.add_argument("inputs", nargs="*", help='mp3 files, folders of songs, and/or "Artist - Title" to find on Soulseek')
    ap.add_argument("-o", "--out", default="drum_samples", help="output folder (default: drum_samples)")
    ap.add_argument("--separator", choices=["demucs", "lalal"], default="demucs",
                    help="demucs = separate on this Mac, free (default); lalal = lalal.ai, uses your minutes")
    ap.add_argument("--api-key", default=os.environ.get("LALAL_API_KEY"), help="lalal.ai license key (or set LALAL_API_KEY)")
    ap.add_argument("--already-drums", action="store_true", help="inputs are already drum stems; skip lalal.ai")
    ap.add_argument("--splitter", choices=["andromeda", "perseus", "orion", "phoenix", "lyra", "lynx"],
                    help="lalal.ai model (default: auto)")
    ap.add_argument("--extraction", choices=["clear_cut", "deep_extraction"], default="clear_cut",
                    help="clear_cut = less bleed from other instruments (default)")
    g = ap.add_argument_group("one-shots")
    g.add_argument("--sensitivity", type=float, default=0.07, help="onset threshold; lower = more hits (default 0.07)")
    g.add_argument("--max-oneshot-len", type=float, default=1.5, help="max one-shot length in seconds (default 1.5)")
    g.add_argument("--max-oneshots", type=int, default=48, help="max one-shots kept per song (default 48)")
    g.add_argument("--oneshot-similarity", type=float, default=0.90,
                   help="0-1; hits more similar than this are treated as duplicates (default 0.90)")
    g.add_argument("--floor-db", type=float, default=-30, help="ignore hits quieter than this vs. song peak (default -30)")
    g = ap.add_argument_group("loops")
    g.add_argument("--loop-bars", type=int, nargs="+", default=[1, 2], help="loop lengths in bars (default: 1 2)")
    g.add_argument("--max-loops", type=int, default=12, help="max loops kept per song per length (default 12)")
    g.add_argument("--loop-similarity", type=float, default=0.85, help="0-1 duplicate threshold for loops (default 0.85)")
    g.add_argument("--loop-floor-db", type=float, default=-30, help="skip loops quieter than this vs. song peak")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false", help="don't peak-normalise exports")
    g = ap.add_argument_group("vocals (Demucs separator)")
    g.add_argument("--no-vocals", dest="vocals", action="store_false", help="don't extract vocals")
    g.add_argument("--max-vocal-phrases", type=int, default=24, help="max vocal phrases per song (default 24)")
    g.add_argument("--vocal-similarity", type=float, default=0.85,
                   help="0-1; phrases more alike than this count as repeats (default 0.85)")
    ap.add_argument("--no-drum-parts", dest="drum_parts", action="store_false",
                    help="don't split drums into kick/snare/toms/hi-hat/ride/crash (DrumSep, ~1 min per song)")
    ap.add_argument("--max-part-oneshots", type=int, default=16, help="max one-shots per drum part per song (default 16)")
    ap.add_argument("--keep-drum-stem", action="store_true",
                    help="keep each song's full drum stem after slicing (default: delete it to save space)")
    ap.add_argument("--keep-part-stems", action="store_true",
                    help="keep the DrumSep part stems after slicing (default: delete them to save space)")
    ap.add_argument("--split-existing", action="store_true",
                    help="only DrumSep + slice the parts of songs already in the library (no other steps)")
    ap.add_argument("--reslice", action="store_true",
                    help="rebuild drum one-shots/loops for songs that already have them (default: keep them)")
    ap.add_argument("--jobs", type=int, default=max(1, min(6, (os.cpu_count() or 4) - 2)),
                    help="songs sliced at once in the Demucs pipeline (default: CPU cores - 2, max 6)")
    ap.add_argument("--no-apple-loops", dest="apple_loops", action="store_false",
                    help="don't also save loops as Apple Loops (~/Library/Audio/Apple Loops/User Loops/SingleFiles/Drum Sampler)")
    g = ap.add_argument_group("saving lalal.ai minutes (--separator lalal only)")
    g.add_argument("--full", action="store_true",
                   help="always send the whole song (skip the local check for songs that are one drum loop)")
    g.add_argument("--excerpt-seconds", type=float, default=60,
                   help="length sent to lalal.ai when the drums are basically one loop (default 60)")
    g.add_argument("--loop-threshold", type=float, default=0.6,
                   help="0-1; how alike a song's least-alike sections must be to count as one loop "
                        "(default 0.6; raise it to send fewer songs as excerpts)")
    g = ap.add_argument_group("soulseek (for \"Artist - Title\" inputs)")
    g.add_argument("--parallel-downloads", type=int, default=3, help="songs downloaded at once (default 3)")
    g.add_argument("--songs-file", help='text file with one "Artist - Title" per line; finished songs are '
                   'removed from it (default with no inputs: songs.txt next to this script)')
    g.add_argument("--slsk-user", default=os.environ.get("SLSK_USERNAME"), help="Soulseek username (or set SLSK_USERNAME)")
    g.add_argument("--slsk-pass", default=os.environ.get("SLSK_PASSWORD"), help="Soulseek password (or set SLSK_PASSWORD)")
    g.add_argument("--download-dir", default="~/Soulseek Downloads/complete/drum_sampler",
                   help="where downloaded mp3s go (default: ~/Soulseek Downloads/complete/drum_sampler)")
    g.add_argument("--slsk-port", type=int, default=2234,
                   help="port other Soulseek users connect to; default 2234 = SoulseekQt's, which your router may already forward")
    g.add_argument("--library", nargs="*", default=["~/Soulseek Downloads"],
                   help="folders checked for an mp3 you already have before downloading (default: ~/Soulseek Downloads)")
    args = ap.parse_args()

    songs, queries = collect_inputs(args.inputs)
    default_list = Path(__file__).resolve().parent / "songs.txt"
    if not args.inputs and not args.songs_file and default_list.exists() and not args.split_existing:
        args.songs_file = str(default_list)
        log(f"No inputs given; using {default_list}")
    songs_file = Path(args.songs_file).expanduser() if args.songs_file else None
    if songs_file:
        queries += [l.strip() for l in songs_file.read_text().splitlines()
                    if l.strip() and not l.lstrip().startswith("#")]
    out = Path(args.out).expanduser()
    use_vocals = args.vocals and args.separator == "demucs" and not args.already_drums
    args.vocals_dir = out.parent / "vocal_samples" if use_vocals else None  # always next to the drum library
    if args.already_drums:
        args.drum_parts = False  # the "stem" is the user's own file; don't write records next to it
    if args.split_existing:
        run_parts_pass(out, args)
        return
    if args.separator == "demucs" and not args.already_drums:
        run_pipeline(songs, queries, songs_file, out, args)
        return

    query_for: dict[Path, str] = {}
    if queries:
        from soulseek import fetch_songs
        download_dir = Path(args.download_dir).expanduser()
        roots = [download_dir] + [Path(r).expanduser() for r in args.library]
        got = fetch_songs(queries, download_dir, roots, args.slsk_user, args.slsk_pass, args.slsk_port)
        songs += [got[q] for q in queries if q in got]
        query_for = {got[q]: q for q in queries if q in got}
    songs = list(dict.fromkeys(songs))
    if not songs:
        sys.exit("No input audio files found.")
    infos = [song_info(s) for s in songs]

    if args.already_drums:
        stems = {s: s for s in songs}
    else:
        if not args.api_key:
            sys.exit("Set LALAL_API_KEY or pass --api-key (or use --already-drums for pre-split stems).")
        client = Lalal(args.api_key, args.splitter, args.extraction)
        left = client.minutes_left()
        if left is not None:
            log(f"lalal.ai minutes left: {left:.1f}")
        stems = extract_drum_stems(infos, out, client, args)

    grand_os = grand_lp = 0
    for info in infos:
        stem = stems.get(info.path)
        if stem is None:
            continue
        if not args.already_drums and bar_grid_for(info, stem, out) is None:
            log(f"[beats] {info.path.name}: no full mix to find bars on; guessing from the drums (less reliable)")
        split_drum_parts(info, stem, out, args)
        _, _, n_os, n_lp, _, _ = slice_song(info, stem, out, args)
        finish_song(info, stem, out, args)
        grand_os += n_os
        grand_lp += n_lp
        q = query_for.get(info.path)
        if songs_file and q and n_os + n_lp > 0:
            remove_from_list(songs_file, q)
            log(f"[list] removed '{q}' from {songs_file.name}")
    log(f"\nDone: {grand_os} one-shots, {grand_lp} loops -> {out}")


def export_part_oneshots(y, sr, song_slug, out_dir: Path, args) -> int:
    """One-shots from a single drum part (e.g. the kick stem): already one kind of drum, so no
    labelling; just hits, de-duplicated, loudest takes kept."""
    hits, mono = find_hits(y, sr, args.sensitivity, args.max_oneshot_len, 0.03, args.floor_db)
    if not hits:
        return 0
    fps = [fingerprint(mono[h.start:h.end], sr, 0.25, 16) for h in hits]
    kept = sorted(dedupe(hits, fps, [h.strength for h in hits], args.oneshot_similarity, args.max_part_oneshots),
                  key=lambda h: h.start)
    for k, h in enumerate(kept, 1):
        seg = fade(y[..., h.start:h.end], sr, 0.5, min(15.0, (h.end - h.start) / sr * 1000 * 0.3))
        if args.normalize:
            seg = normalize(seg)
        write_wav(out_dir / f"{song_slug}_{k:03d}{EXT}", seg, sr)
    return len(kept)


def slice_parts(info: Song, stem: Path, out: Path, args, grid: dict | None) -> int:
    """One-shots and loops from each DrumSep part of a song. Returns the number of files made."""
    import drum_parts

    total = 0
    parts_dir = info.dir(out) / "drum_parts"
    for part in drum_parts.kept_parts(stem) or []:
        src = drum_parts.part_stem(parts_dir, part, info.name)
        if not src.exists():
            continue
        os_dir, lp_dir = parts_dir / part / "oneshots", parts_dir / part / "loops"
        if audio_glob(os_dir, f"{info.name}_[0-9][0-9][0-9]") and not args.reslice:
            continue
        y, sr = librosa.load(src, sr=None, mono=False)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        total += export_part_oneshots(y, sr, info.name, os_dir, args)
        n_lp, _ = export_loops(y, sr, info.name, lp_dir, args, grid, apple_folder=f"{info.artist}/{part}")
        total += n_lp
    return total


def downbeats_saved(stem: Path) -> dict | None:
    import downbeats

    return downbeats.load_grid(stem)


def sliced_marker(stem: Path) -> Path:
    """Written when a song is fully sliced and its drum stem deleted; stands in for the stem so
    the song isn't separated again. Lives in <Artist>/.drum_sampler/ (see records.py)."""
    from records import record

    return record(stem, "sliced")


def finish_song(info: Song, stem: Path, out: Path, args) -> None:
    """Once everything has been cut (one-shots exist; DrumSep ran and its parts were sliced),
    delete the full drum stem and the DrumSep part stems. The bar grid, the parts record and a
    .sliced.json marker stay in <Artist>/.drum_sampler/."""
    import json
    import drum_parts

    if args.already_drums:
        return
    if args.drum_parts and not args.keep_part_stems and drum_parts.kept_parts(stem) is not None \
            and audio_glob(info.dir(out) / "oneshots", f"*/{info.name}_[0-9][0-9][0-9]"):
        freed = 0
        for part in drum_parts.kept_parts(stem):
            ps = drum_parts.part_stem(info.dir(out) / "drum_parts", part, info.name)
            if ps.exists():
                freed += ps.stat().st_size
                ps.unlink()
        if freed:
            log(f"[cleanup] {info.artist} / {info.name}: drum-part stems deleted ({freed / 1e6:.0f} MB freed)")
    if args.keep_drum_stem or not stem.exists():
        return
    if not audio_glob(info.dir(out) / "oneshots", f"*/{info.name}_[0-9][0-9][0-9]"):
        return
    if args.drum_parts:
        if drum_parts.kept_parts(stem) is None:  # DrumSep hasn't run (callers only get here after
            return                               # the parts were sliced successfully)
    size = stem.stat().st_size
    marker = sliced_marker(stem)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"deleted": stem.name, "bytes": size}))
    stem.unlink()
    try:
        stem.parent.rmdir()  # drum_stems/ goes once it's empty
    except OSError:
        pass
    log(f"[cleanup] {info.artist} / {info.name}: drum stem deleted ({size / 1e6:.0f} MB freed)")


def split_drum_parts(info: Song, stem: Path, out: Path, args) -> None:
    """DrumSep: kick/snare/toms/hi-hat/ride/crash stems from the drum stem (GPU; once per song)."""
    if not args.drum_parts:
        return
    import drum_parts

    if drum_parts.kept_parts(stem) is not None:
        return
    if not stem.exists():
        log(f"[drumsep] {info.path.name}: drum stem already deleted; can't split it")
        return
    log(f"[drumsep] {info.path.name}: splitting drums into kick/snare/toms/hi-hat/ride/crash...")
    t0 = time.time()
    try:
        kept = drum_parts.split(stem, info.dir(out) / "drum_parts", info.name)
    except Exception as e:
        log(f"[error] {info.path.name}: DrumSep failed: {e}")
        return
    log(f"[drumsep] {info.path.name}: kept {', '.join(kept) or 'nothing'} ({time.time() - t0:.0f} s)")


def slice_song(info: Song, stem: Path, out: Path, args) -> tuple[str, str, int, int, float, int]:
    """Cut one drum stem into one-shots and loops. Runs in a worker process in the Demucs
    pipeline, so it must only use its arguments (the bar grid is already saved next to the stem)."""
    existing = audio_glob(info.dir(out) / "oneshots", f"*/{info.name}_[0-9][0-9][0-9]")
    if existing and not args.reslice:
        # Already sliced: keep those files exactly as they are (re-slicing gives the same result,
        # but costs time and could swap a near-identical loop). Only vocals get added.
        n_os = len(existing)
        n_lp = len(audio_glob(info.dir(out) / "loops", f"{info.name}_*bpm_*bar_*"))
        grid = downbeats_saved(stem)
        tempo = float(grid["tempo"]) if grid else 0.0
    elif not stem.exists():
        log(f"[slice] {info.artist} / {info.name}: drum stem was deleted after slicing; nothing to re-slice")
        n_os = n_lp = 0
        grid, tempo = downbeats_saved(stem), 0.0
    else:
        y, sr = librosa.load(stem, sr=None, mono=False)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        n_os = export_oneshots(y, sr, info.name, info.dir(out) / "oneshots", args)
        grid = None if args.already_drums else bar_grid_for(info, stem, out)
        n_lp, tempo = export_loops(y, sr, info.name, info.dir(out) / "loops", args, grid)
    n_parts = 0
    if args.drum_parts:
        n_parts = slice_parts(info, stem, out, args, grid if grid else (None if args.already_drums else bar_grid_for(info, stem, out)))
    n_vx = 0
    if args.vocals_dir:
        acapella, phrases = vocal_paths(info, args.vocals_dir)
        existing_vx = audio_glob(phrases, f"{info.name}_phrase_[0-9][0-9][0-9]")
        if existing_vx and not args.reslice:
            n_vx = len(existing_vx)  # already cut; keep them as they are
        elif acapella.exists():
            n_vx = export_vocal_phrases(acapella, info.name, phrases, args)
    log(f"[slice] {info.artist} / {info.name}: {n_os} one-shots, {n_lp} loops" + (f" @ {tempo:g} BPM" if tempo else "")
        + (f", {n_parts} drum-part samples" if n_parts else "") + (f", {n_vx} vocal phrases" if n_vx else ""))
    return info.artist, info.name, n_os, n_lp, tempo, n_vx


def run_parts_pass(out: Path, args) -> None:
    """Existing library only: DrumSep each song that has a drum stem but no parts yet, slice the
    parts, then (unless --keep-drum-stem) delete the drum stem. Nothing else is redone: no
    separation, no vocals, no re-slicing of the full drums."""
    from concurrent.futures import ProcessPoolExecutor

    stems = sorted(p for ext in AUDIO_OUT for p in out.glob(f"*/drum_stems/*{ext}") if ".part" not in p.name)
    log(f"[parts] {len(stems)} drum stems in {out}")
    made = [0]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = []
        for stem in stems:
            info = Song(stem, stem.parent.parent.name, stem.stem)
            split_drum_parts(info, stem, out, args)
            grid = downbeats_saved(stem)

            def done(f, info=info, stem=stem):
                try:
                    n = f.result()
                except Exception as e:
                    log(f"[error] {info.artist} / {info.name}: part slicing failed: {e}")
                    return
                made[0] += n
                log(f"[parts] {info.artist} / {info.name}: {n} drum-part samples")
                finish_song(info, stem, out, args)

            fut = pool.submit(slice_parts, info, stem, out, args, grid)
            fut.add_done_callback(done)
            futures.append(fut)
        log(f"[parts] all songs split; waiting for {sum(not f.done() for f in futures)} to finish slicing")
    log(f"\nDone: {made[0]} drum-part samples")


def run_pipeline(songs: list[Path], queries: list[str], songs_file: Path | None, out: Path, args) -> None:
    """Demucs mode, with the three stages overlapping instead of one after the other:
      1. Soulseek: songs download in a background thread, several at a time, and are handed
         over the moment each one lands (local files first)
      2. GPU (this thread): Demucs separates each song as soon as it arrives, then Beat This
         finds its bars. One at a time: the GPU is the shared resource.
      3. CPU: each separated song is sliced in a pool of worker processes while the GPU moves
         on to the next song."""
    import queue
    import threading
    from concurrent.futures import ProcessPoolExecutor

    ready: queue.Queue = queue.Queue()
    for s in dict.fromkeys(songs):
        ready.put((None, s))
    done_marker = object()

    if queries:
        from soulseek import fetch_songs

        download_dir = Path(args.download_dir).expanduser()
        roots = [download_dir] + [Path(r).expanduser() for r in args.library]

        def fetch() -> None:
            try:
                fetch_songs(queries, download_dir, roots, args.slsk_user, args.slsk_pass, args.slsk_port,
                            parallel=args.parallel_downloads, on_ready=lambda q, p: ready.put((q, p)))
            except Exception as e:
                import traceback
                log(f"[soulseek] download step failed: {e}")
                log(traceback.format_exc())
            finally:
                ready.put(done_marker)

        threading.Thread(target=fetch, daemon=True).start()
    else:
        ready.put(done_marker)

    list_lock = threading.Lock()
    totals = [0, 0, 0]
    seen: set[Path] = set()
    futures = []

    def finished(fut, q, info, stem):
        try:
            _, _, n_os, n_lp, _, n_vx = fut.result()
            finish_song(info, stem, out, args)
        except Exception as e:
            log(f"[error] slicing failed{f' for {q}' if q else ''}: {e}")
            return
        totals[0] += n_os
        totals[1] += n_lp
        totals[2] += n_vx
        if songs_file and q and n_os + n_lp > 0:
            with list_lock:
                remove_from_list(songs_file, q)
            log(f"[list] removed '{q}' from {songs_file.name}")

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        while True:
            item = ready.get()
            if item is done_marker:
                break
            q, path = item
            if path in seen:
                continue
            seen.add(path)
            info = song_info_for(path, q, out)
            stem = demucs_stems([info], out, args.vocals_dir).get(path)
            if stem is None:
                continue
            bar_grid_for(info, stem, out)  # on the GPU here, so the workers only read the saved grid
            split_drum_parts(info, stem, out, args)
            fut = pool.submit(slice_song, info, stem, out, args)
            fut.add_done_callback(lambda f, q=q, i=info, st=stem: finished(f, q, i, st))
            futures.append(fut)
        if not seen:
            sys.exit("No input audio files found.")
        log(f"[pipeline] all songs separated; waiting for {sum(not f.done() for f in futures)} to finish slicing")
    log(f"\nDone: {totals[0]} one-shots, {totals[1]} loops -> {out}"
        + (f"\n      {totals[2]} vocal phrases -> {args.vocals_dir}" if args.vocals_dir else ""))

if __name__ == "__main__":
    main()
